import { useState, useEffect } from 'react';

export function useTypewriter(text: string, speed: number = 38, startDelay: number = 600) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    // Reset state when text changes
    setDisplayed('');
    setDone(false);

    let index = 0;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    timeoutId = setTimeout(() => {
      intervalId = setInterval(() => {
        if (index < text.length) {
          index++;
          setDisplayed(text.substring(0, index));
        } else {
          setDone(true);
          if (intervalId) {
            clearInterval(intervalId);
          }
        }
      }, speed);
    }, startDelay);

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (intervalId) clearInterval(intervalId);
    };
  }, [text, speed, startDelay]);

  return { displayed, done };
}
