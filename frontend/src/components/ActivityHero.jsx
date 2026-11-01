import { AnimatePresence, motion as Motion } from 'framer-motion';
import { AlertCircle, Info } from 'lucide-react';
import { RULE_LABELS, colorFor } from '../constants';

/** Big "current activity" card. Shows when a post-processing rule changed the model's output. */
export default function ActivityHero({ prediction, emptyText }) {
  if (!prediction) {
    return (
      <div className="hero-empty">
        <div className="hero-label muted">No prediction yet</div>
        <p className="muted small">{emptyText}</p>
      </div>
    );
  }
  const { activity, confidence, model_activity, rule } = prediction;
  const pct = Math.round((confidence || 0) * 100);
  const uncertain = activity === 'Uncertain';

  return (
    <div>
      <div className="hero-row">
        <span className="swatch lg" style={{ background: colorFor(activity) }} aria-hidden />
        <AnimatePresence mode="wait">
          <Motion.div
            key={activity}
            className="hero-label"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.18 }}
          >
            {activity}
          </Motion.div>
        </AnimatePresence>
      </div>

      <div className="meter" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
           aria-label="Confidence">
        <div className="meter-fill" style={{ width: `${pct}%`, background: colorFor(activity) }} />
      </div>
      <div className="hero-meta">
        <span>Confidence <strong>{pct}%</strong></span>
        <span className="muted">Model’s top class: {model_activity}</span>
      </div>

      {uncertain && (
        <p className="note warn"><AlertCircle size={14} /> Confidence is below 40%, so no activity is claimed.</p>
      )}
      {rule && (
        <p className="note"><Info size={14} /> {RULE_LABELS[rule] || rule}.</p>
      )}
    </div>
  );
}
