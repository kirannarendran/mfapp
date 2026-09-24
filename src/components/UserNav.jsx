import React, { useEffect, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext';

const UserNav = ({ onOpenProfile }) => {
  const { user, isAuthenticated, loginWithGoogle, logout, authError } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const buttonRef = useRef(null);
  const dropdownRef = useRef(null);

  const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;

  // Initialize and render Google button when not authenticated
  useEffect(() => {
    if (isAuthenticated || !clientId) return;

    let checkCount = 0;
    const maxChecks = 50; // 5 seconds max

    const initGoogleBtn = () => {
      if (window.google?.accounts?.id && buttonRef.current) {
        try {
          window.google.accounts.id.initialize({
            client_id: clientId,
            callback: (response) => {
              if (response?.credential) {
                loginWithGoogle(response.credential).catch(err => {
                  console.error("Google sign-in error:", err);
                });
              }
            },
          });

          // Render Google's official branded button
          buttonRef.current.innerHTML = '';
          window.google.accounts.id.renderButton(buttonRef.current, {
            theme: 'outline',
            size: 'medium',
            type: 'standard',
            shape: 'pill',
            text: 'signin_with',
            logo_alignment: 'left',
          });
        } catch (err) {
          console.error("Error rendering Google button:", err);
        }
      } else if (checkCount < maxChecks) {
        checkCount++;
        setTimeout(initGoogleBtn, 100);
      }
    };

    initGoogleBtn();
  }, [isAuthenticated, clientId, loginWithGoogle]);

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };
    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  if (!isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        {authError && (
          <span className="text-xs text-rose-500 max-w-[150px] truncate" title={authError}>
            {authError}
          </span>
        )}
        <div ref={buttonRef} className="h-9 flex items-center min-w-[140px] justify-end" />
      </div>
    );
  }

  const initials = user?.name
    ? user.name.split(' ').map(n => n[0]).slice(0, 2).join('').toUpperCase()
    : 'U';

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsDropdownOpen(prev => !prev)}
        className="flex items-center gap-2.5 p-1.5 pl-2 pr-3 rounded-full hover:bg-slate-100 transition-colors border border-slate-200/80 focus:outline-none focus:ring-2 focus:ring-finance-primary/20"
        aria-expanded={isDropdownOpen}
        aria-haspopup="true"
      >
        {user?.avatar_url ? (
          <img
            src={user.avatar_url}
            alt={user.name || 'User avatar'}
            className="w-7 h-7 rounded-full object-cover border border-slate-200"
            referrerPolicy="no-referrer"
          />
        ) : (
          <div className="w-7 h-7 rounded-full bg-finance-primary/10 text-finance-primary font-semibold text-xs flex items-center justify-center border border-finance-primary/20">
            {initials}
          </div>
        )}
        <span className="text-sm font-medium text-slate-700 max-w-[120px] md:max-w-[160px] truncate text-left">
          {user?.name || user?.email}
        </span>
        <svg
          className={`w-4 h-4 text-slate-400 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isDropdownOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-2xl shadow-xl border border-slate-200/80 py-2 z-50 animate-in fade-in zoom-in-95 duration-150">
          <div className="px-4 py-3 border-b border-slate-100">
            <p className="text-sm font-semibold text-slate-800 truncate">{user?.name}</p>
            <p className="text-xs text-slate-500 truncate mt-0.5">{user?.email}</p>
            <span className="inline-flex items-center gap-1.5 mt-2 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              Google Connected
            </span>
          </div>

          <div className="p-1 space-y-0.5">
            {onOpenProfile && (
              <button
                onClick={() => {
                  setIsDropdownOpen(false);
                  onOpenProfile();
                }}
                className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 rounded-xl transition-colors font-medium text-left"
              >
                <div className="flex items-center gap-2.5">
                  <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  <span>My Profile</span>
                </div>
                {!user?.profileCompleted && (
                  <span className="w-2 h-2 rounded-full bg-finance-primary animate-pulse" title="Profile incomplete" />
                )}
              </button>
            )}

            <button
              onClick={() => {
                setIsDropdownOpen(false);
                logout();
              }}
              className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-rose-600 hover:bg-rose-50 rounded-xl transition-colors font-medium text-left"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserNav;
