import Me from '../types/Me';

const LOCAL_STORAGE_ME_KEY = 'me';

const getMe = (): Me | null => {
  const storedMe = localStorage.getItem(LOCAL_STORAGE_ME_KEY);
  return storedMe ? JSON.parse(storedMe) : null;
};

const setMe = (user: Me): void => {
  localStorage.setItem(LOCAL_STORAGE_ME_KEY, JSON.stringify(user));
};

const clearMe = (): void => {
  localStorage.removeItem(LOCAL_STORAGE_ME_KEY);
};

const LocalStorage = {
  getMe,
  setMe,
  clearMe,
};

export default LocalStorage;
