const initializeTheme = () => {
  try {
    const themeStorageString = localStorage.getItem('theme-storage');
    if (themeStorageString) {
      const themeStorage = JSON.parse(themeStorageString);
      if (themeStorage.state?.theme) {
        return themeStorage.state.theme;
      }
    }
  } catch (error) {
    console.error('Failed to initialize theme from localStorage:', error);
  }
  return 'light';
};

const initialTheme = initializeTheme();

export default initialTheme;
