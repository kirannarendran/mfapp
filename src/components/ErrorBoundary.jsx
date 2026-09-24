import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[ErrorBoundary caught error]:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card p-8 text-center max-w-lg mx-auto my-12 bg-white border border-rose-100 shadow-sm rounded-2xl">
          <div className="w-12 h-12 rounded-full bg-rose-50 text-rose-500 mx-auto flex items-center justify-center mb-4 text-xl font-bold">
            ⚠️
          </div>
          <h3 className="text-lg font-bold text-slate-800 mb-2">Something went wrong</h3>
          <p className="text-sm text-slate-500 mb-6 leading-relaxed">
            {this.props.fallbackMessage || "An unexpected error occurred while rendering this section. You can return to the fund list or reload the application."}
          </p>
          <div className="flex justify-center gap-3">
            {this.props.onReset && (
              <button
                type="button"
                onClick={() => {
                  this.setState({ hasError: false, error: null });
                  this.props.onReset();
                }}
                className="btn-primary text-sm px-5 py-2.5"
              >
                Return to Overview
              </button>
            )}
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="px-4 py-2 border border-slate-200 text-slate-700 text-sm font-medium rounded-xl hover:bg-slate-50 transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
