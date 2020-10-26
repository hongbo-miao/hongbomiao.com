const convertToNumber = (value: string): number | null => {
  if (['unknown', 'n/a'].indexOf(value) !== -1) {
    return null;
  }

  // Remove digit grouping
  const numberString = value.replace(/,/, '');
  return Number(numberString);
};

export default convertToNumber;
