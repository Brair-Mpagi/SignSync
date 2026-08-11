/**
 * Confidence and warning display.
 *
 * Plan §16.3 requires in-product transparency about limits, and plan §14 lists
 * community trust as the highest-impact risk. These components exist so that
 * "show the confidence" is a shared, reviewable piece of UI rather than something
 * each screen re-decides — and so that removing it is a visible deletion.
 */

import type { Warning } from '../api';

const SEVERE = new Set([
  'low_confidence',
  'unrecognised_signs',
  'missing_signs',
  'untranslated_words',
]);

export function WarningList({ warnings }: { warnings: Warning[] }) {
  if (warnings.length === 0) return null;
  return (
    <ul className="warnings">
      {warnings.map((warning) => (
        <li
          key={warning.code}
          className={SEVERE.has(warning.code) ? 'warning warning--severe' : 'warning'}
        >
          {warning.message}
        </li>
      ))}
    </ul>
  );
}

export function ConfidenceMeter({
  confidence,
  needsRepeat,
}: {
  confidence: number;
  needsRepeat: boolean;
}) {
  const percent = Math.round(confidence * 100);
  const level = percent < 60 ? 'low' : percent < 80 ? 'medium' : 'high';

  return (
    <div className="confidence">
      <div className="confidence-bar" role="meter" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}>
        <span className={level} style={{ width: `${percent}%` }} />
      </div>
      <span className="confidence-label">
        confidence {percent}%
        {needsRepeat && ' — ask the signer to repeat'}
      </span>
    </div>
  );
}

/**
 * Never hide this. Plan §16.3 positions the system as an assistive tool and not a
 * replacement for a certified interpreter in medical, legal or safety-critical
 * settings, and a dismissible disclaimer is not a disclaimer.
 */
export function Disclaimer({ text }: { text: string }) {
  return (
    <p className="disclaimer" role="note">
      {text}
    </p>
  );
}
