import './LoadingState.css';

const BARS = [
  { dur: '0.8s', max: '32px' },
  { dur: '0.6s', max: '48px' },
  { dur: '1.0s', max: '24px' },
  { dur: '0.7s', max: '44px' },
  { dur: '0.9s', max: '36px' },
];

function LoadingState() {
  return (
    <div className="loading-state" aria-live="polite" aria-label="Loading recommendations">
      <div className="loading-eq" aria-hidden="true">
        {BARS.map((b, i) => (
          <div key={i} className="loading-bar" style={{ '--dur': b.dur, '--max': b.max }} />
        ))}
      </div>
      <p className="loading-text">Crunching the audio data…</p>
      <div className="skeleton-list" aria-hidden="true">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="skeleton-card" style={{ animationDelay: `${i * 0.1}s` }} />
        ))}
      </div>
    </div>
  );
}

export default LoadingState;
