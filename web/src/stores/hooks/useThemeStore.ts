import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import initialTheme from '../theme/constants/initialTheme';
import applyTheme from '../theme/utils/applyTheme';
import type { ThemeState } from '@/stores/theme/types/ThemeState';

const useThemeStore = create<ThemeState>()(
  persist(
    (set) => ({
      theme: initialTheme,
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

export default useThemeStore;
