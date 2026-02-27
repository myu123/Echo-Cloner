/**
 * Main App component with routing
 */
import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { Upload, Brain, Wand2, Folder, Cpu, Menu, X } from 'lucide-react';
import { UploadPage, TrainingPage, InferencePage, ModelsPage } from '@/pages';
import { systemAPI } from '@/services/api';

function App() {
  const [cudaAvailable, setCudaAvailable] = useState<boolean | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    checkCuda();
  }, []);

  const checkCuda = async () => {
    try {
      const health = await systemAPI.healthCheck();
      setCudaAvailable(health.cuda_available);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const navigation = [
    { name: 'Upload', path: '/upload', icon: Upload },
    { name: 'Train', path: '/train', icon: Brain },
    { name: 'Generate', path: '/generate', icon: Wand2 },
    { name: 'Models', path: '/models', icon: Folder },
  ];

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-dark-bg text-dark-text">
        {/* Header */}
        <header className="bg-dark-surface border-b border-dark-border sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                  <Wand2 className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-dark-text">Echo Cloner</h1>
                  <p className="text-xs text-dark-muted">XTTS-v2</p>
                </div>
              </div>

              {/* Desktop Navigation */}
              <nav className="hidden md:flex items-center gap-2">
                {navigation.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                        isActive
                          ? 'bg-primary-600 text-white'
                          : 'text-dark-muted hover:bg-dark-border hover:text-dark-text'
                      }`
                    }
                  >
                    <item.icon className="w-4 h-4" />
                    {item.name}
                  </NavLink>
                ))}
              </nav>

              {/* Status & Mobile Menu Button */}
              <div className="flex items-center gap-3">
                {/* CUDA Status */}
                <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-dark-border rounded-full">
                  <Cpu className="w-4 h-4" />
                  <span className="text-xs font-medium">
                    {cudaAvailable === null ? (
                      'Checking...'
                    ) : cudaAvailable ? (
                      <span className="text-green-500">CUDA Ready</span>
                    ) : (
                      <span className="text-red-500">No CUDA</span>
                    )}
                  </span>
                </div>

                {/* Mobile Menu Button */}
                <button
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                  className="md:hidden p-2 text-dark-muted hover:text-dark-text"
                >
                  {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                </button>
              </div>
            </div>

            {/* Mobile Navigation */}
            {mobileMenuOpen && (
              <div className="md:hidden border-t border-dark-border py-4">
                <nav className="flex flex-col gap-2">
                  {navigation.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={({ isActive }) =>
                        `flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition-colors ${
                          isActive
                            ? 'bg-primary-600 text-white'
                            : 'text-dark-muted hover:bg-dark-border hover:text-dark-text'
                        }`
                      }
                    >
                      <item.icon className="w-5 h-5" />
                      {item.name}
                    </NavLink>
                  ))}
                </nav>
              </div>
            )}
          </div>
        </header>

        {/* CUDA Warning Banner */}
        {cudaAvailable === false && (
          <div className="bg-red-500/10 border-b border-red-500">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
              <p className="text-sm text-red-500 text-center">
                ⚠ CUDA not available - Training will not work. Please ensure PyTorch with CUDA support is installed.
              </p>
            </div>
          </div>
        )}

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/train" element={<TrainingPage />} />
            <Route path="/generate" element={<InferencePage />} />
            <Route path="/models" element={<ModelsPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="mt-auto border-t border-dark-border">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <p className="text-center text-sm text-dark-muted">
              Echo Cloner • XTTS-v2 • Built with React + FastAPI
            </p>
          </div>
        </footer>

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1e293b',
              color: '#e2e8f0',
              border: '1px solid #334155',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#1e293b',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#1e293b',
              },
            },
          }}
        />
      </div>
    </BrowserRouter>
  );
}

export default App;
