/**
 * The three modes of plan §18.3.
 *
 * Mode A  sign to speech
 * Mode B  speech to sign
 * Mode C  conversation — both, with turn-taking
 *
 * Capabilities come from /health and features the server lacks are *disabled*, not
 * hidden behind a button that silently does nothing.
 */

import { useCallback, useEffect, useState } from 'react';

import {
  api,
  type AnimationData,
  type EnglishToSignResponse,
  type Health,
  type Rig,
  type SignToEnglishResponse,
  type Warning,
} from './api';
import { Avatar } from './components/Avatar';
import { ConfidenceMeter, Disclaimer, WarningList } from './components/Warnings';

type Mode = 'sign-to-speech' | 'speech-to-sign' | 'conversation';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [rig, setRig] = useState<Rig | null>(null);
  const [mode, setMode] = useState<Mode>('speech-to-sign');
  const [error, setError] = useState<string | null>(null);

  const [english, setEnglish] = useState('Where is the hospital?');
  const [signed, setSigned] = useState<EnglishToSignResponse | null>(null);
  const [animation, setAnimation] = useState<AnimationData | null>(null);
  const [nowSigning, setNowSigning] = useState<string | null>(null);

  const [glossInput, setGlossInput] = useState('ME NEED HELP');
  const [markers, setMarkers] = useState<string[]>([]);
  const [spoken, setSpoken] = useState<SignToEnglishResponse | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.rig()])
      .then(([h, r]) => {
        setHealth(h);
        setRig(r);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const handleSpeakToSign = useCallback(async () => {
    try {
      const result = await api.englishToSign(english);
      setSigned(result);
      setAnimation(result.animation);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [english]);

  const handleSignToSpeech = useCallback(async () => {
    try {
      const glosses = glossInput.trim().toUpperCase().split(/\s+/).filter(Boolean);
      setSpoken(await api.signToEnglish(glosses, markers));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [glossInput, markers]);

  const toggleMarker = (marker: string) =>
    setMarkers((current) =>
      current.includes(marker) ? current.filter((m) => m !== marker) : [...current, marker],
    );

  const deploymentWarnings: Warning[] = health?.warnings ?? [];

  return (
    <div className="app">
      <header>
        <h1>SignSync</h1>
        <p>Ugandan Sign Language ↔ English</p>
        {health && (
          <div className="capabilities">
            {Object.entries(health.capabilities).map(([name, ready]) => (
              <span key={name} className={ready ? 'cap cap--on' : 'cap cap--off'}>
                {name.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        )}
      </header>

      {health && <Disclaimer text={health.disclaimer} />}
      {error && <p className="error">{error}</p>}

      <nav className="modes">
        {(['speech-to-sign', 'sign-to-speech', 'conversation'] as Mode[]).map((value) => (
          <button
            key={value}
            className={mode === value ? 'active' : ''}
            onClick={() => setMode(value)}
            // Conversation mode needs both directions live; offering it without a
            // recogniser would be a button that does nothing.
            disabled={value === 'conversation' && !health?.capabilities.recognition}
          >
            {value.replace(/-/g, ' ')}
          </button>
        ))}
      </nav>

      {(mode === 'speech-to-sign' || mode === 'conversation') && (
        <section className="panel">
          <h2>English to sign</h2>
          <div className="row">
            <input value={english} onChange={(e) => setEnglish(e.target.value)} />
            <button onClick={handleSpeakToSign}>Sign it</button>
          </div>

          <Avatar rig={rig} animation={animation} onGlossChange={setNowSigning} />
          <p className="now-signing">{nowSigning ?? '—'}</p>

          {signed && (
            <>
              <pre className="notation">{signed.notation}</pre>
              {signed.missing.length > 0 && (
                <p className="note">No sign available for: {signed.missing.join(', ')}</p>
              )}
              {signed.generated.length > 0 && (
                <p className="note">
                  Generated motion (not a recording of a signer): {signed.generated.join(', ')}
                </p>
              )}
              <WarningList warnings={signed.warnings} />
            </>
          )}
        </section>
      )}

      {(mode === 'sign-to-speech' || mode === 'conversation') && (
        <section className="panel">
          <h2>Sign to English</h2>
          <div className="row">
            <input value={glossInput} onChange={(e) => setGlossInput(e.target.value)} />
            <button onClick={handleSignToSpeech}>Translate</button>
          </div>

          <fieldset className="markers">
            <legend>Non-manual markers</legend>
            {['head_shake', 'brow_raise', 'brow_furrow'].map((marker) => (
              <label key={marker}>
                <input
                  type="checkbox"
                  checked={markers.includes(marker)}
                  onChange={() => toggleMarker(marker)}
                />
                {marker.replace(/_/g, ' ')}
              </label>
            ))}
          </fieldset>

          {spoken && (
            <>
              <output className="english">{spoken.text || '—'}</output>
              <ConfidenceMeter confidence={spoken.confidence} needsRepeat={spoken.needsRepeat} />
              {!spoken.speech.audible && (
                <p className="note">No audio: {spoken.speech.detail}</p>
              )}
              <p className="trace">{spoken.frame}</p>
              <WarningList warnings={spoken.warnings} />
            </>
          )}
        </section>
      )}

      <WarningList warnings={deploymentWarnings} />
    </div>
  );
}
