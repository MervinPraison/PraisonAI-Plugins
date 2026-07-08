/**
 * PraisonAI-Plugins gate configuration.
 */

module.exports = {
  repoFullName: 'MervinPraison/PraisonAI-Plugins',
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
  agentPyChecks: false,
  reviewBotLogins: [
    'coderabbitai[bot]',
    'qodo-code-review[bot]',
    'greptile-apps[bot]',
  ],
};
