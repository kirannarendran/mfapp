import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

const PROFESSIONS = [
  'Software & Technology / IT',
  'Banking, Finance & Accounting',
  'Business Owner / Entrepreneur',
  'Healthcare, Medical & Pharma',
  'Sales, Marketing & Consulting',
  'Engineering & Operations',
  'Government & Public Sector',
  'Legal & Education',
  'Student',
  'Retired',
  'Other'
];

const EXPERIENCE_LEVELS = [
  { id: 'Beginner', title: 'Beginner', desc: 'New to mutual funds & SIPs' },
  { id: 'Intermediate', title: 'Intermediate', desc: '1 – 3 years investing' },
  { id: 'Experienced', title: 'Experienced', desc: 'Comfortable with risk metrics' }
];

const ProfileModal = ({ isOpen = true, onClose, onBack, onCompleteOnboarding, mode = 'modal' }) => {
  const { user, updateProfile } = useAuth();

  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [age, setAge] = useState('');
  const [profession, setProfession] = useState('');
  const [investmentExperience, setInvestmentExperience] = useState('Intermediate');

  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Pre-fill state when modal opens or user changes
  useEffect(() => {
    if (user && (isOpen || mode === 'page')) {
      setFirstName(user.firstName || (user.name ? user.name.split(' ')[0] : ''));
      setLastName(user.lastName || (user.name ? user.name.split(' ').slice(1).join(' ') : ''));
      setAge(user.age || '');
      setProfession(user.profession || '');
      setInvestmentExperience(user.investmentExperience || 'Intermediate');
      setError(null);
      setSaveSuccess(false);
    }
  }, [user, isOpen, mode]);

  // Handle ESC key (only in modal mode)
  useEffect(() => {
    if (mode !== 'modal') return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen && onClose) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, mode]);

  if (!user) return null;
  if (mode === 'modal' && !isOpen) return null;

  const isFirstTime = !user.profileCompleted;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    setError(null);

    try {
      const profileData = {
        firstName: firstName.trim(),
        lastName: lastName.trim(),
        age: age ? parseInt(age, 10) : null,
        profession,
        investmentExperience
      };

      await updateProfile(profileData);

      const email = (user?.email || '').toLowerCase();
      if (email) {
        localStorage.setItem(`fundsense_profile_completed_${email}`, 'true');
        localStorage.setItem(`fundsense_profile_data_${email}`, JSON.stringify(profileData));
      }
      if (user?.id) {
        localStorage.setItem(`fundsense_profile_completed_${user.id}`, 'true');
      }
      localStorage.setItem('fundsense_profile_completed_global', 'true');
      localStorage.setItem('fundsense_profile_completed', 'true');

      setSaveSuccess(true);
      if (mode === 'modal') {
        setTimeout(() => {
          if (onCompleteOnboarding) {
            onCompleteOnboarding();
          } else if (onClose) {
            onClose();
          }
        }, 700);
      }
    } catch (err) {
      setError(err.message || 'Failed to update profile');
    } finally {
      setIsSaving(false);
    }
  };

  const formContent = (
    <form onSubmit={handleSubmit} className="space-y-5">
      {error && (
        <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 font-medium">
          {error}
        </div>
      )}

      {saveSuccess && (
        <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-700 font-medium flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-600 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
          </svg>
          <span>Profile updated successfully! Your preferences are now synchronized.</span>
        </div>
      )}

      {/* Name Fields */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            First Name <span className="text-rose-500">*</span>
          </label>
          <input
            type="text"
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            required
            className="w-full px-3.5 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:outline-none focus:ring-2 focus:ring-finance-primary/20 focus:border-finance-primary transition-all"
            placeholder="First name"
          />
        </div>
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Last Name
          </label>
          <input
            type="text"
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            className="w-full px-3.5 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:outline-none focus:ring-2 focus:ring-finance-primary/20 focus:border-finance-primary transition-all"
            placeholder="Last name"
          />
        </div>
      </div>

      {/* Email (Read-only) */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="block text-xs font-semibold text-slate-700">
            Email Address
          </label>
          <span className="text-[10px] font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200/60 flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
            Verified
          </span>
        </div>
        <input
          type="email"
          value={user.email}
          disabled
          className="w-full px-3.5 py-2.5 text-sm bg-slate-100/70 border border-slate-200 rounded-xl text-slate-500 cursor-not-allowed"
        />
      </div>

      {/* Age & Profession Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="sm:col-span-1">
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Age
          </label>
          <input
            type="number"
            min="18"
            max="100"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            placeholder="e.g. 28"
            className="w-full px-3.5 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:outline-none focus:ring-2 focus:ring-finance-primary/20 focus:border-finance-primary transition-all"
          />
        </div>
        <div className="sm:col-span-2">
          <label className="block text-xs font-semibold text-slate-700 mb-1.5">
            Profession / Industry
          </label>
          <select
            value={profession}
            onChange={(e) => setProfession(e.target.value)}
            className="w-full px-3 py-2.5 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:bg-white focus:outline-none focus:ring-2 focus:ring-finance-primary/20 focus:border-finance-primary transition-all text-slate-800"
          >
            <option value="">Select your profession</option>
            {PROFESSIONS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Investment Experience */}
      <div>
        <label className="block text-xs font-semibold text-slate-700 mb-2">
          Investment Experience
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
          {EXPERIENCE_LEVELS.map((lvl) => (
            <button
              type="button"
              key={lvl.id}
              onClick={() => setInvestmentExperience(lvl.id)}
              className={`p-3 rounded-xl border text-left transition-all ${
                investmentExperience === lvl.id
                  ? 'border-finance-primary bg-finance-primary/5 ring-1 ring-finance-primary text-slate-900 shadow-sm'
                  : 'border-slate-200 bg-slate-50 hover:bg-slate-100/70 text-slate-600'
              }`}
            >
              <p className="text-xs font-bold">{lvl.title}</p>
              <p className="text-[10px] text-slate-500 mt-0.5 leading-snug">{lvl.desc}</p>
            </button>
          ))}
        </div>
      </div>


      {/* Privacy Consent Notice (shown on first time modal) */}
      {isFirstTime && (
        <div className="p-3.5 bg-blue-50/70 border border-blue-100 rounded-xl">
          <p className="text-[11px] text-slate-600 leading-relaxed">
            <svg className="w-3.5 h-3.5 inline-block mr-1 text-blue-500 -mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            By saving your profile, you agree that FundSense.AI may store your demographic and investing preferences for personalizing portfolio diagnostics. No financial account or broker credentials are collected.
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="pt-4 border-t border-slate-100 flex items-center justify-end gap-3">
        {mode === 'modal' && (
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2.5 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition-colors"
          >
            {isFirstTime ? 'Do this later' : 'Cancel'}
          </button>
        )}
        <button
          type="submit"
          disabled={isSaving}
          className="px-6 py-2.5 rounded-xl bg-finance-primary hover:bg-finance-primary-dark text-white font-bold text-xs shadow-sm hover:shadow transition-all disabled:opacity-50 flex items-center gap-2 cursor-pointer"
        >
          {isSaving ? (
            <>
              <div className="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
              <span>Saving...</span>
            </>
          ) : (
            <span>{isFirstTime ? 'Agree & Save Profile' : 'Save Profile Changes'}</span>
          )}
        </button>
      </div>
    </form>
  );

  // In-page view (rendered when user navigates to "My Profile" tab from sidebar or dropdown)
  if (mode === 'page') {
    return (
      <div className="max-w-4xl mx-auto w-full pb-16 animate-fade-in">
        {onBack && (
          <button
            onClick={onBack}
            className="mb-6 text-sm text-finance-primary hover:text-finance-primary-dark flex items-center gap-2 font-medium cursor-pointer"
          >
            ← Back to Fund Universe
          </button>
        )}

        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden mb-6">
          {/* Page Top Banner */}
          <div className="p-6 sm:p-8 bg-gradient-to-r from-slate-900 to-slate-800 text-white flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">
            <div className="flex items-center gap-4">
              {user.avatarUrl ? (
                <img
                  src={user.avatarUrl}
                  alt={user.name}
                  className="w-16 h-16 rounded-2xl object-cover border-2 border-white/20 shadow-md"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <div className="w-16 h-16 rounded-2xl bg-white/10 text-white font-black text-2xl flex items-center justify-center border border-white/20 shadow-md">
                  {firstName ? firstName[0].toUpperCase() : 'U'}
                </div>
              )}
              <div>
                <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
                  {user.name || `${firstName} ${lastName}`}
                </h1>
                <p className="text-xs text-slate-300 mt-1">{user.email}</p>
              </div>
            </div>
          </div>

          {/* Form Content */}
          <div className="p-6 sm:p-8">
            <div className="mb-6 pb-4 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-slate-900">My Profile</h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Update your personal information and investing preferences.
                </p>
              </div>
              <span className="text-xs text-slate-400 font-medium hidden sm:inline-block">
                All fields stored securely
              </span>
            </div>

            {formContent}
          </div>
        </div>

        {/* Security and Privacy Assurance */}
        <div className="bg-slate-50 rounded-2xl p-5 border border-slate-200/80 text-xs text-slate-600 flex items-start gap-3">
          <svg className="w-5 h-5 text-finance-primary shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <p className="leading-relaxed">
            <strong className="text-slate-800">Your Privacy is Guaranteed:</strong> FundSense.AI never accesses your bank account or broker portfolios. Your profile details are used strictly to tailor mathematical constraints and risk models within this educational application.
          </p>
        </div>
      </div>
    );
  }

  // Modal mode (overlay popup used only during first-time initial login)
  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 animate-in fade-in duration-200">
      <div className="relative w-full max-w-lg bg-white rounded-3xl shadow-2xl border border-slate-200/90 overflow-hidden">
        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-slate-100 flex items-start justify-between">
          <div className="flex items-center gap-3.5">
            {user.avatarUrl ? (
              <img
                src={user.avatarUrl}
                alt={user.name}
                className="w-12 h-12 rounded-2xl object-cover border border-slate-200 shadow-sm"
                referrerPolicy="no-referrer"
              />
            ) : (
              <div className="w-12 h-12 rounded-2xl bg-finance-primary/10 text-finance-primary font-bold text-lg flex items-center justify-center border border-finance-primary/20 shadow-sm">
                {firstName ? firstName[0].toUpperCase() : 'U'}
              </div>
            )}
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                {isFirstTime ? `Welcome, ${firstName || 'Investor'}!` : 'My Profile'}
              </h2>
              <p className="text-xs text-slate-500">
                {isFirstTime ? 'Complete your details for personalized risk analytics' : 'Manage your personal details and investing preferences'}
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors cursor-pointer"
            title="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form Body */}
        <div className="p-6">
          {formContent}
        </div>
      </div>
    </div>
  );
};

export default ProfileModal;
