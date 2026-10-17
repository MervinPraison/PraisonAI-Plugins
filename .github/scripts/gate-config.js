/**
 * PraisonAI-Plugins gate configuration.
 */

module.exports = {
  repoFullName: 'MervinPraison/PraisonAI-Plugins',
  gitUser: 'MervinPraison',
  gitEmail: '454862+MervinPraison@users.noreply.github.com',
  testCommand: 'pytest tests/ -q',
  productPathPrefixes: ['src/praisonai_plugins/', 'tests/'],
  sensitivePathPatterns: [
    /^\.github\/workflows\//,
    /^pyproject\.toml$/,
  ],
  requiredCheckPatterns: [/^ci$/i, /python/i, /test/i, /lint/i, /ruff/i],
  ciWorkflowFile: 'ci.yml',
  ciWorkflowName: 'CI',
  mergeGateWorkflowRuns: ['CI', 'Claude Assistant'],
  ciFailureWorkflowRuns: ['CI'],
  pypiPackageName: 'praisonai-plugins',
  packagePaths: ['src/praisonai_plugins', 'pyproject.toml'],
  finalClaudeScope:
    'SCOPE: Focus ONLY on PraisonAI-Plugins (src/praisonai_plugins, tests). '
    + 'Lifecycle plugins use praisonai.plugins; sandbox backends use praisonai.sandbox. '
    + 'Do NOT add agent-callable tools here — those belong in PraisonAI-Tools.',
  finalClaudeProductValue:
    '4. Product value: plugins wrap execution lifecycle (hooks, guardrails, policies); '
    + 'reject scope creep into praisonaiagents core or agent tools.',
  mergeGateProductValue:
    'Confirm the plugin change strengthens execution-lifecycle extensions (hooks, guardrails, '
    + 'policies, sandbox backends) without scope creep; BLOCK duplicated capabilities or unused knobs.',
  mergeGateLayering:
    'Repo routing: BLOCK agent-callable tools (belong in PraisonAI-Tools) and core SDK changes '
    + '(belong in praisonaiagents).',
  agentPyChecks: false,
  reviewBotLogins: [
    'coderabbitai[bot]',
    'qodo-code-review[bot]',
    'greptile-apps[bot]',
  ],
};
