module.exports = {
  hooks: {
    'pre-commit': 'yarn lint && yarn lint:css',
    'pre-push': 'yarn test',
  }
}
