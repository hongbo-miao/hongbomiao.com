import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ThemeState } from '@/stores/types/ThemeState';

const applyTheme = (theme: 'light' | 'dark') => {
  const root = document.documentElement;
  if (theme === 'dark') {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
};

export const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      theme: 'light',
      setTheme: (theme) => {
        set({ theme });
        applyTheme(theme);
      },
    }),
    {
      name: 'theme-storage',
      onRehydrateStorage: (state) => {
        if (state?.theme) {
          applyTheme(state.theme);
        }
      },
    },
  ),
);
