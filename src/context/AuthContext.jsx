import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { trackEvent } from '../utils/analytics';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem('fundsense_token'));
  const [isLoading, setIsLoading] = useState(true);
  const [authError, setAuthError] = useState(null);

  // Helper to rehydrate profile across server deployments
  const processUserProfile = useCallback((userData, activeToken) => {
    if (!userData) return null;
    const email = (userData.email || '').toLowerCase();
    
    // Look up saved profile data across all fallback keys
    const savedDataStr = (email && localStorage.getItem(`fundsense_profile_data_${email}`)) ||
      localStorage.getItem('fundsense_profile_data_global') ||
      localStorage.getItem('fundsense_profile_data') ||
      localStorage.getItem('fundsense_user_profile');

    const isLocallyCompleted = Boolean(
      userData.profileCompleted ||
      (email && localStorage.getItem(`fundsense_profile_completed_${email}`) === 'true') ||
      (userData.id && localStorage.getItem(`fundsense_profile_completed_${userData.id}`) === 'true') ||
      localStorage.getItem('fundsense_profile_completed_global') === 'true' ||
      localStorage.getItem('fundsense_profile_completed') === 'true' ||
      savedDataStr
    );

    if (isLocallyCompleted) {
      userData.profileCompleted = true;
      if (savedDataStr) {
        try {
          const savedData = JSON.parse(savedDataStr);
          if (!userData.profession && savedData.profession) userData.profession = savedData.profession;
          if (!userData.age && savedData.age) userData.age = savedData.age;
          if (!userData.investmentExperience && savedData.investmentExperience) userData.investmentExperience = savedData.investmentExperience;
          if (!userData.firstName && savedData.firstName) userData.firstName = savedData.firstName;
          if (!userData.lastName && savedData.lastName) userData.lastName = savedData.lastName;

          // Re-sync to backend in background if server database was freshly provisioned
          if (activeToken) {
            fetch('/api/auth/profile', {
              method: 'PUT',
              headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${activeToken}`,
              },
              body: JSON.stringify(savedData),
            }).catch(() => {});
          }
        } catch (e) {
          console.warn('Failed to parse cached profile data:', e);
        }
      }
    }
    return userData;
  }, []);

  // Restore session on app load
  useEffect(() => {
    const restoreSession = async () => {
      const savedToken = localStorage.getItem('fundsense_token');
      if (!savedToken) {
        setIsLoading(false);
        return;
      }

      try {
        const response = await fetch('/api/auth/me', {
          headers: {
            Authorization: `Bearer ${savedToken}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          const processedUser = processUserProfile(data.user, savedToken);
          setUser(processedUser);
          setToken(savedToken);
        } else {
          // Token expired or invalid
          localStorage.removeItem('fundsense_token');
          setUser(null);
          setToken(null);
        }
      } catch (err) {
        console.error('[AuthContext] Failed to restore session:', err);
      } finally {
        setIsLoading(false);
      }
    };

    restoreSession();
  }, [processUserProfile]);

  const loginWithGoogle = useCallback(async (credential) => {
    setIsLoading(true);
    setAuthError(null);
    try {
      const response = await fetch('/api/auth/google', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ credential }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Google login failed');
      }

      localStorage.setItem('fundsense_token', data.token);
      setToken(data.token);

      const processedUser = processUserProfile(data.user, data.token);
      setUser(processedUser);
      trackEvent('login', { method: 'google' });
      return processedUser;
    } catch (err) {
      console.error('[AuthContext] Login error:', err);
      setAuthError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [processUserProfile]);

  const updateProfile = useCallback(async (profileData) => {
    const activeToken = token || localStorage.getItem('fundsense_token');
    if (!activeToken) throw new Error('Not authenticated');

    const response = await fetch('/api/auth/profile', {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${activeToken}`,
      },
      body: JSON.stringify(profileData),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Failed to update profile');
    }

    const email = (data.user?.email || '').toLowerCase();
    if (email) {
      localStorage.setItem(`fundsense_profile_completed_${email}`, 'true');
      localStorage.setItem(`fundsense_profile_data_${email}`, JSON.stringify(profileData));
    }
    if (data.user?.id) {
      localStorage.setItem(`fundsense_profile_completed_${data.user.id}`, 'true');
    }
    localStorage.setItem('fundsense_profile_completed_global', 'true');
    localStorage.setItem('fundsense_profile_completed', 'true');
    localStorage.setItem('fundsense_profile_data_global', JSON.stringify(profileData));
    localStorage.setItem('fundsense_profile_data', JSON.stringify(profileData));
    localStorage.setItem('fundsense_user_profile', JSON.stringify(profileData));

    data.user.profileCompleted = true;
    setUser(data.user);
    trackEvent('profile_update', { 
      profession: profileData.profession || 'not_specified',
      experience: profileData.investmentExperience || 'not_specified'
    });
    return data.user;
  }, [token]);

  const logout = useCallback(() => {
    trackEvent('logout');
    localStorage.removeItem('fundsense_token');
    setUser(null);
    setToken(null);
    setAuthError(null);
    if (window.google?.accounts?.id) {
      window.google.accounts.id.disableAutoSelect();
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        authError,
        isAuthenticated: !!user,
        loginWithGoogle,
        updateProfile,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
