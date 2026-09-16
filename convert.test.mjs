import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  extractText,
  normalizeRole,
  normalizeChatRequest,
  convertResponsesToChat,
  looksLikeResponsesRequest,
  normalizeIncoming,
} from './convert.mjs';

test('developer role becomes system', () => {
  const out = normalizeChatRequest({
    messages: [
      { role: 'developer', content: [{ type: 'input_text', text: 'be terse' }] },
      { role: 'user', content: 'hi' },
    ],
  });
  assert.equal(out.messages[0].role, 'system');
  assert.equal(out.messages[0].content, 'be terse');
  assert.equal(out.messages[1].role, 'user');
  assert.equal(out.messages[1].content, 'hi');
});

test('unknown role is coerced to user, not passed through', () => {
  const out = normalizeChatRequest({
    messages: [{ role: 'narrator', content: 'once upon a time' }],
  });
  assert.equal(out.messages[0].role, 'user');
  assert.equal(out.messages[0].content, 'once upon a time');
});

test('Responses input_text parts flatten to chat messages', () => {
  const out = convertResponsesToChat({
    model: 'deepseek/deepseek-v4.1-flash',
    instructions: '',
    input: [
      { type: 'message', role: 'developer', content: [{ type: 'input_text', text: 'sys' }] },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hello' }] },
    ],
    tools: [{ type: 'function', name: 'read', description: 'Read a file', parameters: { type: 'object' } }],
    stream: true,
  });
  assert.equal(out.model, 'deepseek/deepseek-v4.1-flash');
  assert.equal(out.stream, true);
  assert.deepEqual(out.messages.map((m) => m.role), ['system', 'user']);
  assert.equal(out.messages[0].content, 'sys');
  assert.equal(out.messages[1].content[0].type, 'text');
  assert.equal(out.messages[1].content[0].text, 'hello');
  assert.equal(out.tools[0].function.name, 'read');
});

test('looksLikeResponsesRequest distinguishes chat vs responses', () => {
  assert.equal(looksLikeResponsesRequest({ messages: [{ role: 'user', content: 'a' }] }), false);
  assert.equal(looksLikeResponsesRequest({ input: [{ type: 'message', role: 'user', content: 'a' }] }), true);
});

test('normalizeIncoming accepts a Responses body on the chat path', () => {
  const out = normalizeIncoming({
    model: 'x',
    input: [{ type: 'message', role: 'user', content: [{ type: 'input_text', text: 'ping' }] }],
  });
  assert.equal(out.messages[0].role, 'user');
  assert.equal(extractText(out.messages[0].content), 'ping');
});

test('function_call items become assistant tool_calls', () => {
  const out = convertResponsesToChat({
    input: [
      { type: 'function_call', call_id: 'c1', name: 'read', arguments: '{"path":"a"}' },
      { type: 'function_call_output', call_id: 'c1', output: 'file contents' },
    ],
  });
  assert.equal(out.messages[0].role, 'assistant');
  assert.equal(out.messages[0].tool_calls[0].function.name, 'read');
  assert.equal(out.messages[1].role, 'tool');
  assert.equal(out.messages[1].content, 'file contents');
});

test('normalizeRole maps developer/instruction to system', () => {
  assert.equal(normalizeRole('developer'), 'system');
  assert.equal(normalizeRole('instruction'), 'system');
  assert.equal(normalizeRole('model'), 'assistant');
});
