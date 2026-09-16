/**
 * Normalize Chat Completions / Responses payloads into the Chat shape
 * that buildCcRequest already understands, and that Command Code accepts.
 *
 * CC params.messages only allow role user|assistant|tool, and content must
 * be an array of parts. OpenAI Chat / Codex Responses still send developer
 * + string content / input_text parts.
 */

export function extractText(content) {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => extractText(part)).filter(Boolean).join('\n');
  }
  if (typeof content !== 'object') return String(content);
  if (typeof content.text === 'string') return content.text;
  if (typeof content.input_text === 'string') return content.input_text;
  if (typeof content.output_text === 'string') return content.output_text;
  if (typeof content.value === 'string') return content.value;
  return '';
}

export function normalizeRole(role) {
  switch (role) {
    case 'system':
    case 'developer':
    case 'instruction':
      return 'system';
    case 'assistant':
    case 'model':
      return 'assistant';
    case 'tool':
    case 'function':
      return 'tool';
    case 'user':
      return 'user';
    default:
      return 'user';
  }
}

function mapContentPart(part) {
  if (part == null) return null;
  if (typeof part === 'string') return { type: 'text', text: part };
  if (typeof part !== 'object') return { type: 'text', text: String(part) };
  if (part.type === 'image_url' || part.type === 'image') return part;
  if (part.type === 'input_text' || part.type === 'output_text' || part.type === 'text') {
    return { type: 'text', text: part.text || '' };
  }
  if (typeof part.text === 'string' && !part.type) {
    return { type: 'text', text: part.text };
  }
  return part;
}

export function flattenUserContent(content) {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map(mapContentPart).filter(Boolean);
  return extractText(content);
}

export function flattenAssistantContent(content) {
  if (content == null) return content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const parts = content.map(mapContentPart).filter((p) => p && (p.type === 'text' ? p.text : true));
    return parts;
  }
  return extractText(content) || null;
}

export function looksLikeResponsesRequest(body) {
  if (!body || typeof body !== 'object') return false;
  if (Array.isArray(body.messages) && body.messages.length > 0 && !Array.isArray(body.input)) {
    return false;
  }
  return Array.isArray(body.input) || (typeof body.instructions === 'string' && body.instructions.length > 0);
}

function mapResponsesTools(tools) {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  return tools.map((t) => {
    if (!t || typeof t !== 'object') return t;
    if (t.function && t.function.name) return t;
    if (t.type === 'function' || t.name) {
      return {
        type: 'function',
        function: {
          name: t.name || t.function?.name || '',
          description: t.description || t.function?.description || '',
          parameters: t.parameters || t.function?.parameters || { type: 'object', properties: {} },
        },
      };
    }
    return t;
  });
}

export function normalizeChatRequest(req) {
  if (!req || typeof req !== 'object') return { messages: [] };
  const src = Array.isArray(req.messages) ? req.messages : [];
  const messages = [];
  for (const msg of src) {
    if (!msg || typeof msg !== 'object') continue;
    const role = normalizeRole(msg.role);
    if (role === 'system') {
      messages.push({ role: 'system', content: extractText(msg.content) });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: msg.tool_call_id || msg.toolCallId || msg.id,
        name: msg.name,
        content: extractText(msg.content),
      });
      continue;
    }
    if (role === 'assistant') {
      const next = { role: 'assistant', content: flattenAssistantContent(msg.content) };
      if (msg.tool_calls) next.tool_calls = msg.tool_calls;
      messages.push(next);
      continue;
    }
    messages.push({ role: 'user', content: flattenUserContent(msg.content) });
  }
  return { ...req, messages };
}

export function convertResponsesToChat(req) {
  if (!req || typeof req !== 'object') return { messages: [] };
  const messages = [];
  const instructions = extractText(req.instructions);
  if (instructions) messages.push({ role: 'system', content: instructions });

  const input = Array.isArray(req.input) ? req.input : [];
  for (const item of input) {
    if (typeof item === 'string') {
      messages.push({ role: 'user', content: item });
      continue;
    }
    if (!item || typeof item !== 'object') continue;
    const type = item.type || 'message';

    if (type === 'function_call') {
      messages.push({
        role: 'assistant',
        content: null,
        tool_calls: [{
          id: item.call_id || item.id,
          type: 'function',
          function: {
            name: item.name || '',
            arguments: typeof item.arguments === 'string'
              ? item.arguments
              : JSON.stringify(item.arguments || {}),
          },
        }],
      });
      continue;
    }
    if (type === 'function_call_output') {
      messages.push({
        role: 'tool',
        tool_call_id: item.call_id || item.id,
        content: extractText(item.output ?? item.content),
      });
      continue;
    }
    if (type === 'reasoning' || type === 'item_reference') continue;

    const role = normalizeRole(item.role || (type === 'message' ? 'user' : 'user'));
    if (role === 'system') {
      messages.push({ role: 'system', content: extractText(item.content) });
    } else if (role === 'assistant') {
      messages.push({ role: 'assistant', content: flattenAssistantContent(item.content) });
    } else if (role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: item.call_id || item.tool_call_id || item.id,
        content: extractText(item.content ?? item.output),
      });
    } else {
      messages.push({ role: 'user', content: flattenUserContent(item.content) });
    }
  }

  const normalized = normalizeChatRequest({ messages });
  const out = {
    model: req.model,
    messages: normalized.messages,
    stream: req.stream === true,
  };
  if (req.max_output_tokens != null) out.max_tokens = req.max_output_tokens;
  else if (req.max_tokens != null) out.max_tokens = req.max_tokens;
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.parallel_tool_calls !== undefined) out.parallel_tool_calls = req.parallel_tool_calls;
  if (req.tool_choice !== undefined) out.tool_choice = req.tool_choice;
  const effort = req.reasoning_effort || req.reasoning?.effort;
  if (effort !== undefined) out.reasoning_effort = effort;
  const tools = mapResponsesTools(req.tools);
  if (tools) out.tools = tools;
  return out;
}

export function normalizeIncoming(body) {
  if (looksLikeResponsesRequest(body)) return convertResponsesToChat(body);
  return normalizeChatRequest(body);
}
