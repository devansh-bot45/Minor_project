import React, { useState } from 'react';
import { useGoogleLogin } from '@react-oauth/google';
import { motion as Motion } from 'framer-motion';
import { Activity, Loader2 } from 'lucide-react';
import axios from 'axios';

const Login = ({ backendUrl, setAuthToken, setUser }) => {
  const [loading, setLoading] = useState(false);

  const login = useGoogleLogin({
    onSuccess: async (tokenResponse) => {
      try {
        setLoading(true);
        // We use the robust access_token approach to bypass Cross-Origin-Opener-Policy errors
        const res = await axios.post(`${backendUrl}/api/auth/google`, {
          access_token: tokenResponse.access_token
        });
        const { token, user } = res.data;
        setAuthToken(token);
        setUser(user);
        localStorage.setItem('har_auth_token', token);
        localStorage.setItem('har_user', JSON.stringify(user));
      } catch (err) {
        console.error('Login failed:', err);
        alert('Login failed. Ensure backend is running and Google Client ID is configured.');
      } finally {
        setLoading(false);
      }
    },
    onError: (error) => {
      console.error('Google Login Failed', error);
      alert('Google Login Failed: ' + (error?.error || 'Unknown error'));
    }
  });

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
      <Motion.div 
        className="card" 
        style={{ maxWidth: '400px', width: '100%', textAlign: 'center', padding: '40px 24px' }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Activity size={48} color="var(--primary-color)" style={{ marginBottom: '16px' }} />
        <h2 style={{ justifyContent: 'center', fontSize: '1.5rem', marginBottom: '8px' }}>Sign In</h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '32px', fontSize: '0.9rem' }}>
          Authenticate to track your daily activities and generate AI health reports.
        </p>
        
        <div style={{ display: 'flex', justifyContent: 'center' }}>
          <button 
            className="btn-start" 
            onClick={() => login()} 
            disabled={loading}
            style={{ width: 'auto', padding: '12px 24px', fontSize: '1rem', background: '#fff', color: '#000', border: 'none', borderRadius: '4px', fontWeight: 'bold' }}
          >
            {loading ? <Loader2 className="spin" size={20} /> : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <img src="https://www.svgrepo.com/show/475656/google-color.svg" alt="Google" style={{ width: '20px', height: '20px' }} />
                Sign in with Google
              </div>
            )}
          </button>
        </div>
      </Motion.div>
    </div>
  );
};

export default Login;
