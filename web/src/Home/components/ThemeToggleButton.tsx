import { Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useThemeStore } from '@/stores/themeStore';

export function ThemeToggleButton() {
  const { theme, setTheme } = useThemeStore();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Check the actual DOM state to determine if dark mode is active
    const checkDarkMode = () => {
      const hasDarkClass = document.documentElement.classList.contains('dark');
      setIsDark(hasDarkClass);
    };

    // Initial check
    checkDarkMode();

    // Create observer to watch for class changes on document element
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('dark');
    } else {
      setTheme('light');
    }
  };

  return (
    <button
      onClick={toggleTheme}
      className="fixed top-4 right-4 z-50 p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-all duration-200 cursor-pointer"
      title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
    >
      {isDark ? (
        <Sun className="w-6 h-6 text-gray-700 dark:text-gray-300" />
      ) : (
        <Moon className="w-6 h-6 text-gray-700 dark:text-gray-300" />
      )}
    </button>
  );
}
