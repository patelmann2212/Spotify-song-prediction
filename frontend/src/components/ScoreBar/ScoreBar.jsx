import './ScoreBar.css';

/**
 * ScoreBar — a thin animated progress bar to represent a 0-1 score.
 * @param {number} value  - score 0.0 to 1.0
 * @param {string} label  - e.g. "Match"
 */
function ScoreBar({ value, label = 'Match' }) {
  if (value == null) return null;
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="score-bar-wrap" title={`${label}: ${pct}%`}>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="score-bar-label">{pct}%</span>
    </div>
  );
}

export default ScoreBar;
