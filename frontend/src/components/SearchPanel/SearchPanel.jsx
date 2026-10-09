import { useState, useRef, useCallback, useEffect } from 'react';
import './SearchPanel.css';

const ENGINES = [
  {
    id: 'weighted',
    icon: '⚡',
    label: 'Hybrid',
    desc: 'Our smartest engine: 40% audio + 20% artist + 15% genre + 15% cluster + 10% popularity.',
    needsSong: true,
  },
  {
    id: 'content',
    icon: '🎵',
    label: 'Audio',
    desc: 'Finds songs with similar acoustic features — energy, tempo, valence, and more.',
    needsSong: true,
  },
  {
    id: 'genre',
    icon: '🎸',
    label: 'Genre',
    desc: 'Recommends within the same genre using audio similarity.',
    needsSong: true,
  },
  {
    id: 'artist',
    icon: '👤',
    label: 'Artist',
    desc: 'Discovers artists who share credits and collaborations with your chosen artist.',
    needsSong: false,
  },
  {
    id: 'mood',
    icon: '🎭',
    label: 'Mood',
    desc: 'Pick a vibe — the engine finds top tracks classified by energy, tempo, and valence.',
    needsSong: false,
  },
];

const MOODS = [
  { id: 'Party', icon: '🎉', cls: 'mood-party' },
  { id: 'Chill', icon: '😌', cls: 'mood-chill' },
  { id: 'Happy', icon: '😊', cls: 'mood-happy' },
  { id: 'Sad',   icon: '💙', cls: 'mood-sad'   },
];

function SearchPanel({ onSearch, onSongSearch, isLoading }) {
  const [engine, setEngine] = useState('weighted');
  const [song, setSong] = useState('');
  const [artist, setArtist] = useState('');
  const [mood, setMood] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSug, setShowSug] = useState(false);
  const debounceRef = useRef(null);
  const wrapRef = useRef(null);

  const activeEngine = ENGINES.find((e) => e.id === engine);

  /* Close suggestions on outside click */
  useEffect(() => {
    function onClickOut(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setShowSug(false);
      }
    }
    document.addEventListener('mousedown', onClickOut);
    return () => document.removeEventListener('mousedown', onClickOut);
  }, []);

  const handleSongChange = useCallback(
    (val) => {
      setSong(val);
      clearTimeout(debounceRef.current);
      if (val.length < 2) { setSuggestions([]); setShowSug(false); return; }
      debounceRef.current = setTimeout(async () => {
        const results = await onSongSearch(val);
        setSuggestions(results);
        setShowSug(results.length > 0);
      }, 300);
    },
    [onSongSearch]
  );

  const pickSuggestion = (item) => {
    setSong(item.track_name);
    setArtist(item.artists || '');
    setSuggestions([]);
    setShowSug(false);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch({ engine, song, artist, mood });
  };

  const isDisabled =
    isLoading ||
    (engine === 'mood' && !mood) ||
    (engine === 'artist' && !artist.trim()) ||
    (activeEngine.needsSong && !song.trim());

  return (
    <div className="search-panel">
      {/* Engine Tabs */}
      <div className="engine-tabs" role="tablist" aria-label="Recommendation engines">
        {ENGINES.map((eng) => (
          <button
            key={eng.id}
            role="tab"
            aria-selected={engine === eng.id}
            className={`engine-tab ${engine === eng.id ? 'active' : ''}`}
            onClick={() => { setEngine(eng.id); setSuggestions([]); setShowSug(false); }}
            type="button"
          >
            <span className="engine-tab-icon">{eng.icon}</span>
            <span className="engine-tab-label">{eng.label}</span>
          </button>
        ))}
      </div>

      <div className="search-body">
        <p className="engine-description">{activeEngine.desc}</p>

        <form onSubmit={handleSubmit}>
          {/* MOOD ENGINE */}
          {engine === 'mood' && (
            <div className="mood-grid" role="radiogroup" aria-label="Select mood">
              {MOODS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={mood === m.id}
                  className={`mood-btn ${m.cls} ${mood === m.id ? 'active' : ''}`}
                  onClick={() => setMood(m.id)}
                >
                  <span className="mood-icon">{m.icon}</span>
                  <span className="mood-label">{m.id}</span>
                </button>
              ))}
            </div>
          )}

          {/* ARTIST ENGINE */}
          {engine === 'artist' && (
            <div className="input-row">
              <div className="input-group">
                <label className="input-label" htmlFor="artist-input">Artist name</label>
                <div className="input-wrap">
                  <span className="input-icon">👤</span>
                  <input
                    id="artist-input"
                    className="text-input"
                    type="text"
                    placeholder="e.g. The Weeknd"
                    value={artist}
                    onChange={(e) => setArtist(e.target.value)}
                    required
                    autoComplete="off"
                  />
                </div>
              </div>
            </div>
          )}

          {/* SONG-BASED ENGINES */}
          {activeEngine.needsSong && (
            <div className="input-row">
              <div className="input-group flex-2" ref={wrapRef}>
                <label className="input-label" htmlFor="song-input">Song title</label>
                <div className="input-wrap">
                  <span className="input-icon">🔍</span>
                  <input
                    id="song-input"
                    className="text-input"
                    type="text"
                    placeholder="e.g. Blinding Lights"
                    value={song}
                    onChange={(e) => handleSongChange(e.target.value)}
                    onFocus={() => suggestions.length > 0 && setShowSug(true)}
                    required
                    autoComplete="off"
                  />
                </div>
                {showSug && suggestions.length > 0 && (
                  <ul className="autocomplete-list" role="listbox">
                    {suggestions.map((s, i) => (
                      <li key={i} role="option">
                        <button
                          type="button"
                          className="autocomplete-item"
                          onClick={() => pickSuggestion(s)}
                        >
                          <span className="autocomplete-track">{s.track_name}</span>
                          <span className="autocomplete-artist">{s.artists}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="input-group">
                <label className="input-label" htmlFor="artist-opt-input">Artist (optional)</label>
                <div className="input-wrap">
                  <span className="input-icon">👤</span>
                  <input
                    id="artist-opt-input"
                    className="text-input"
                    type="text"
                    placeholder="Narrow by artist"
                    value={artist}
                    onChange={(e) => setArtist(e.target.value)}
                    autoComplete="off"
                  />
                </div>
              </div>
            </div>
          )}

          <button type="submit" className="search-submit" disabled={isDisabled}>
            {isLoading && <span className="btn-spinner" aria-hidden="true" />}
            {isLoading ? 'Searching…' : 'Explore'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default SearchPanel;
