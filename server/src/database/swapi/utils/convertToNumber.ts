const convertToNumber = (val: string): number | null => {
  if (['unknown', 'n/a'].indexOf(val) !== -1) {
    return null;
  }

  // Remove digit grouping
  const numberString = val.replace(/,/, '');
  return Number(numberString);
};

export default convertToNumber;
