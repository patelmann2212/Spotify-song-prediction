import { useState } from 'react';
import './App.css';
import Header from './components/Header/Header';
import Hero from './components/Hero/Hero';
import SearchPanel from './components/SearchPanel/SearchPanel';
import ResultsSection from './components/ResultsSection/ResultsSection';
import LoadingState from './components/LoadingState/LoadingState';

const API = 'http://127.0.0.1:8000';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [queryInfo, setQueryInfo] = useState(null);

  const handleSearch = async (formData) => {
    setLoading(true);
    setError(null);
    setHasSearched(true);
    setResults([]);
    setQueryInfo(formData);

    try {
      const response = await fetch(`${API}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, top_n: 8 }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Recommendation request failed.');
      }

      setResults(data.results || []);
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? 'Cannot reach the backend server. Make sure it is running on port 8000.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSongSearch = async (q) => {
    if (!q || q.length < 2) return [];
    try {
      const r = await fetch(`${API}/search?q=${encodeURIComponent(q)}&limit=6`);
      if (!r.ok) return [];
      const d = await r.json();
      return d.results || [];
    } catch {
      return [];
    }
  };

  return (
    <div className="page-wrap">
      <Header />
      <main>
        <Hero />
        <SearchPanel
          onSearch={handleSearch}
          onSongSearch={handleSongSearch}
          isLoading={loading}
        />

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-banner-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {loading && <LoadingState />}

        {!loading && hasSearched && !error && (
          <ResultsSection results={results} queryInfo={queryInfo} />
        )}
      </main>
    </div>
  );
}

export default App;
