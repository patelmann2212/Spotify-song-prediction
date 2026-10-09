import './Hero.css';

const EQ_BARS = [
  { dur: '0.7s', maxH: '16px' },
  { dur: '0.9s', maxH: '32px' },
  { dur: '0.6s', maxH: '24px' },
  { dur: '1.1s', maxH: '36px' },
  { dur: '0.8s', maxH: '20px' },
  { dur: '0.5s', maxH: '28px' },
  { dur: '0.95s', maxH: '14px' },
  { dur: '0.75s', maxH: '36px' },
  { dur: '0.65s', maxH: '22px' },
];

const ENGINES = [
  { icon: '🎵', label: 'Audio Match' },
  { icon: '🎸', label: 'By Genre' },
  { icon: '👤', label: 'By Artist' },
  { icon: '🎭', label: 'By Mood' },
  { icon: '⚡', label: 'Hybrid' },
];

function Hero() {
  return (
    <section className="hero" aria-label="Hero section">
      <div className="hero-bg" aria-hidden="true">
        <div className="blob blob-1" />
        <div className="blob blob-2" />
      </div>

      <div className="hero-content">
        <div className="hero-eyebrow">
          <span>♪</span> Intelligent Music Discovery
        </div>

        <h1 className="hero-title">
          <span className="hero-title-plain">Find Songs That</span>
          <span className="hero-title-accent">Move You</span>
        </h1>

        <p className="hero-subtitle">
          Five AI-powered recommendation engines analyze audio DNA, genre profiles,
          artist similarity, and mood — all from your actual listening catalogue.
        </p>

        <div className="equalizer" aria-hidden="true">
          {EQ_BARS.map((b, i) => (
            <div
              key={i}
              className="eq-bar"
              style={{ '--dur': b.dur, '--max-h': b.maxH }}
            />
          ))}
        </div>

        <div className="engine-pills" aria-label="Available recommendation engines">
          {ENGINES.map((e) => (
            <div key={e.label} className="engine-pill">
              <span>{e.icon}</span>
              <span>{e.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Hero;
