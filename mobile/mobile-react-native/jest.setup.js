// https://github.com/expo/expo/issues/36831#issuecomment-3107047371
// eslint-disable-next-line no-undef
jest.mock('expo/src/winter/ImportMetaRegistry', () => ({
  ImportMetaRegistry: {
    get url() {
      return null;
    },
  },
}));

// eslint-disable-next-line no-undef
if (typeof global.structuredClone === 'undefined') {
  // eslint-disable-next-line no-undef
  global.structuredClone = (object) => JSON.parse(JSON.stringify(object));
}
