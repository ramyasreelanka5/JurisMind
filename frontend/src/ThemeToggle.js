import React from 'react';
import './ThemeToggle.css';

const ThemeToggle = ({ theme, toggleTheme }) => {
  return (
    <button 
      onClick={toggleTheme} 
      className="theme-toggle-btn" 
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      {theme === 'dark' ? (
        <span role="img" aria-label="sun icon" className="icon">☀️</span>
      ) : (
        <span role="img" aria-label="moon icon" className="icon">🌙</span>
      )}
    </button>
  );
};

export default ThemeToggle;