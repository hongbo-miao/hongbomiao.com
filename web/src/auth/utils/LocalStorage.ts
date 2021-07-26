import LocalStorageMe from '../types/LocalStorageMe';

const ME_LOCALSTORAGE_KEY = 'me';

const getMe = (): LocalStorageMe | null => {
  const storedUser = localStorage.getItem(ME_LOCALSTORAGE_KEY);
  return storedUser ? JSON.parse(storedUser) : null;
};

const setMe = (user: LocalStorageMe): void => {
  localStorage.setItem(ME_LOCALSTORAGE_KEY, JSON.stringify(user));
};

const clearMe = (): void => {
  localStorage.removeItem(ME_LOCALSTORAGE_KEY);
};

const LocalStorage = {
  getMe,
  setMe,
  clearMe,
};

export default LocalStorage;
