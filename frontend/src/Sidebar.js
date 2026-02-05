import React from 'react';
import ThemeToggle from './ThemeToggle';
import './Sidebar.css';

const Sidebar = ({ 
  isOpen, 
  onClose, 
  conversations, 
  currentConversationId, 
  onSelectConversation, 
  onNewConversation, 
  onDeleteConversation,
  onLogout,
  user,
  isLoading,
  theme,
  toggleTheme
}) => {

  const handleLogout = () => {
    if (window.confirm('Are you sure you want to log out?')) {
      onLogout();
    }
  };

  const getUserInitial = () => {
    return user?.username ? user.username.charAt(0).toUpperCase() : 'U';
  };

  const sortedConversations = [...conversations].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));

  return (
    <>
      <div 
        className={`sidebar-overlay ${isOpen ? 'show' : ''}`} 
        onClick={onClose}
      />
      
      <div className={`sidebar ${isOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-top">
          <div className="user-info" onClick={handleLogout} title="Click to log out">
            <div className="user-avatar">{getUserInitial()}</div>
            <span className="username">{user?.username || 'User'}</span>
          </div>
          <button className="close-btn" onClick={onClose} title="Close sidebar">✕</button>
        </div>
        
        <div className="conversations-list">
          <div className="list-header">Projects</div>
          {isLoading ? (
            <div className="loading-conversations">Loading...</div>
          ) : sortedConversations.length === 0 ? (
            <div className="no-conversations">No projects yet.</div>
          ) : (
            sortedConversations.map(conv => (
              <div 
                key={conv.id}
                className={`conversation-item ${conv.id === currentConversationId ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <span className="conversation-title"># {conv.title || 'Untitled'}</span>
                <button 
                  className="delete-btn"
                  onClick={(e) => { e.stopPropagation(); onDeleteConversation(conv.id); }}
                  title={`Delete "${conv.title || 'conversation'}"`}
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>
        
        <div className="sidebar-footer">
          <button 
            className="new-conversation-btn" 
            onClick={onNewConversation}
            title="Start a new project"
          >
            <span className="icon">➕</span>
            Create a New Project
          </button>
          <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
        </div>
      </div>
    </>
  );
};

export default Sidebar;