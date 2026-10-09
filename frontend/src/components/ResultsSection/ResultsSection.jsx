import './ResultsSection.css';
import SongCard from '../SongCard/SongCard';

const ENGINE_LABELS = {
  content:  { icon: '🎵', label: 'Audio Similarity' },
  genre:    { icon: '🎸', label: 'Genre-Based' },
  artist:   { icon: '👤', label: 'Artist Similarity' },
  mood:     { icon: '🎭', label: 'Mood-Based' },
  weighted: { icon: '⚡', label: 'Hybrid' },
};

function ResultsSection({ results, queryInfo }) {
  const engine = queryInfo?.engine || 'content';
  const meta = ENGINE_LABELS[engine] || { icon: '🎵', label: engine };

  const title =
    engine === 'artist'
      ? `Artists similar to "${queryInfo?.artist || '?'}"`
      : engine === 'mood'
      ? `Top ${queryInfo?.mood || ''} tracks`
      : `Because you searched for "${queryInfo?.song || '?'}"`;

  if (!results || results.length === 0) {
    return (
      <section className="results-section">
        <div className="empty-state">
          <div className="empty-state-icon">🔇</div>
          <p>No results found. Try a different song or engine.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="results-section" aria-label="Recommendation results">
      <div className="results-header">
        <h2 className="results-title">
          <span>{meta.icon}</span>
          <span>{title}</span>
          <span className="results-badge">{results.length}</span>
        </h2>
        <span className="results-engine-tag">{meta.label}</span>
      </div>

      <ul className="results-list">
        {results.map((result, i) => (
          <li key={i}>
            <SongCard result={result} rank={i + 1} engine={engine} />
          </li>
        ))}
      </ul>

      <p className="results-note">
        Recommendations from the local dataset — no external sources.
      </p>
    </section>
  );
}

export default ResultsSection;
