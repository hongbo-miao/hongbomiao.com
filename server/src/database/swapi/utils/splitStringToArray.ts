const splitStringToArray = (val: string): string[] => {
  return val.split(',').map((s) => s.trim());
};

export default splitStringToArray;
