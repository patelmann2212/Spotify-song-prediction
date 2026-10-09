import './SongCard.css';
import ScoreBar from '../ScoreBar/ScoreBar';

/* Deterministic gradient palette — pick based on char code of first letter */
const GRADIENTS = [
  'linear-gradient(135deg,#1db954,#006b2d)',
  'linear-gradient(135deg,#7c3aed,#4c1d95)',
  'linear-gradient(135deg,#e11d48,#9f1239)',
  'linear-gradient(135deg,#0ea5e9,#0c4a6e)',
  'linear-gradient(135deg,#f59e0b,#92400e)',
  'linear-gradient(135deg,#06b6d4,#164e63)',
  'linear-gradient(135deg,#ec4899,#701a75)',
  'linear-gradient(135deg,#84cc16,#365314)',
];

function getGradient(seed = '') {
  const code = (seed.charCodeAt(0) || 0) + (seed.charCodeAt(1) || 0);
  return GRADIENTS[code % GRADIENTS.length];
}

/* Popularity: 0–100 → 0–10 dots */
function PopDots({ value }) {
  if (value == null || value === '') return null;
  const filled = Math.round((value / 100) * 10);
  return (
    <div className="popularity-dots" title={`Popularity: ${value}/100`}>
      {Array.from({ length: 10 }).map((_, i) => (
        <div key={i} className={`pop-dot ${i < filled ? 'filled' : ''}`} />
      ))}
    </div>
  );
}

function SongCard({ result, rank, engine }) {
  const isArtist = engine === 'artist';

  /* Score to display */
  const score =
    result.final_score != null
      ? result.final_score
      : result.similarity_score != null
      ? result.similarity_score
      : null;

  const scoreLabel =
    engine === 'weighted'
      ? 'Hybrid'
      : engine === 'artist'
      ? 'Match'
      : 'Audio Match';

  const title = isArtist
    ? result.artist || result.genre || '—'
    : result.track_name || '—';

  const subtitle = isArtist ? '' : result.artists || '';
  const artSeed  = title;

  const animDelay = `${(rank - 1) * 0.04}s`;

  return (
    <div
      className={`song-card ${isArtist ? 'artist-card' : ''}`}
      style={{ animationDelay: animDelay }}
    >
      {/* Rank */}
      <div className="card-rank">{rank}</div>

      {/* Artwork tile */}
      <div className="card-art">
        <div
          className="card-art-gradient"
          style={{ background: getGradient(artSeed) }}
          aria-hidden="true"
        >
          {isArtist ? '👤' : title.charAt(0).toUpperCase()}
        </div>
      </div>

      {/* Track info */}
      <div className="card-info">
        <div className="card-title" title={title}>{title}</div>
        {subtitle && <div className="card-artist" title={subtitle}>{subtitle}</div>}

        <div className="card-tags">
          {result.track_genre && !isArtist && (
            <span className="card-tag">{result.track_genre}</span>
          )}
        </div>

        {!isArtist && result.popularity != null && result.popularity !== '' && (
          <PopDots value={result.popularity} />
        )}
      </div>

      {/* Score bar */}
      {score != null && (
        <div className="card-score">
          <ScoreBar value={score} label={scoreLabel} />
        </div>
      )}
    </div>
  );
}

export default SongCard;
