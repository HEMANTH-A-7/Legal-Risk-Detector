import React, { useState, useEffect, useRef } from 'react';
import { useTypewriter } from './hooks/useTypewriter';

interface DetectorDistribution {
  transformer: number;
  ml: number;
  keyword: number;
}

interface CCICMetadata {
  mode: string;
  transformer_threshold: number;
  ml_threshold: number;
  temperature: number;
  detector_distribution: DetectorDistribution;
}

interface SummaryStatistics {
  total_sentences: number;
  risky_count: number;
  risk_percentage: number;
  severity_distribution: Record<string, number>;
}

interface RiskClause {
  original_sentence: string;
  risk_type: string;
  severity: 'High' | 'Medium' | 'Low';
  severity_score: number;
  confidence: number;
  detector: 'transformer' | 'ml' | 'keyword';
  explanation?: string;
}

interface AnalysisResponse {
  risks: RiskClause[];
  llm_available: boolean;
  ccic: CCICMetadata;
  summary: SummaryStatistics;
  error?: string;
  traceback?: string;
}

interface GitHubProfile {
  name: string;
  login: string;
  avatar_url: string;
  html_url: string;
  bio: string;
  public_repos: number;
  followers: number;
}

export const App: React.FC = () => {
  // Input states
  const [file, setFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState('');
  
  // App UI states
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [showCTA, setShowCTA] = useState(false);
  
  // Explanation caching & states
  const [expandedClauses, setExpandedClauses] = useState<Record<number, boolean>>({});
  const [explanations, setExplanations] = useState<Record<number, string>>({});
  const [loadingExplanations, setLoadingExplanations] = useState<Record<number, boolean>>({});

  // Dynamic GitHub user details
  const [githubProfile, setGithubProfile] = useState<GitHubProfile | null>(null);

  // Smooth LERP animation variables for scales
  const [tilt, setTilt] = useState(0);
  const targetTiltRef = useRef<number>(0);
  const currentTiltRef = useRef<number>(0);
  const isHoveredRef = useRef<boolean>(false);

  const analyzerSectionRef = useRef<HTMLDivElement>(null);
  const contactSectionRef = useRef<HTMLDivElement>(null);

  // Typewriter hook with legal-focused text
  const typewriterText = "Analyze contracts for hidden liabilities, risk variables, and compliance anomalies instantly using confidence-calibrated Legal-BERT.";
  const { displayed: typedText, done: typewriterDone } = useTypewriter(typewriterText, 25, 400);

  // 400ms delay for action CTA button
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowCTA(true);
    }, 400);
    return () => clearTimeout(timer);
  }, []);

  // Fetch GitHub profile details
  useEffect(() => {
    fetch('https://api.github.com/users/HEMANTH-A-7')
      .then(res => res.json())
      .then(data => {
        if (data && data.login) {
          setGithubProfile({
            name: data.name || 'Hemanth Amarthi',
            login: data.login,
            avatar_url: data.avatar_url,
            html_url: data.html_url,
            bio: data.bio || 'AI / ML Developer & Researcher',
            public_repos: data.public_repos,
            followers: data.followers
          });
        }
      })
      .catch(err => console.error('Error fetching github profile:', err));
  }, []);

  // requestAnimationFrame loop to LERP the tilt angle seamlessly
  useEffect(() => {
    let animId: number;
    
    const updateAnimation = () => {
      // If not hovered, oscillate slowly using a sine wave
      if (!isHoveredRef.current) {
        const time = Date.now() * 0.0015;
        targetTiltRef.current = Math.sin(time * 1.8) * 3.5; // slow, gentle sway
      }

      const diff = targetTiltRef.current - currentTiltRef.current;
      currentTiltRef.current += diff * 0.085; // smooth easing multiplier
      setTilt(currentTiltRef.current);
      
      animId = requestAnimationFrame(updateAnimation);
    };

    animId = requestAnimationFrame(updateAnimation);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Scale SVG mouse movement handler (local coordinates distance from pivot)
  const handleScaleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    isHoveredRef.current = true;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const relativeX = e.clientX - centerX;
    
    const halfWidth = rect.width / 2;
    const ratio = relativeX / halfWidth; // -1 to +1
    
    // Proportional tilt (clamped between -22 and 22 degrees)
    // Moving mouse to the right pulls the right pan down (clockwise / positive tilt)
    targetTiltRef.current = ratio * 22;
  };

  const handleScaleMouseLeave = () => {
    isHoveredRef.current = false;
  };

  // Scroll utilities
  const scrollToSection = (ref: React.RefObject<HTMLDivElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Handle form submission
  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file && !textInput.trim()) {
      alert('Please upload a file or paste contract text to analyze.');
      return;
    }

    setIsLoading(true);
    setAnalysisResult(null);
    setExpandedClauses({});
    setExplanations({});
    setLoadingExplanations({});

    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else {
      formData.append('text', textInput);
    }

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      setIsLoading(false);
      if (data.error) {
        const detail = data.traceback 
          ? `\n\nDetails:\n${data.traceback.split('\n').slice(-5).join('\n')}` 
          : '';
        alert('Server Error: ' + data.error + detail);
        return;
      }

      setAnalysisResult(data);
      // Wait for DOM layout, then scroll to results
      setTimeout(() => {
        const resultsEl = document.getElementById('results-dashboard');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } catch (err: any) {
      setIsLoading(false);
      alert('Network/Server error: ' + err.message);
    }
  };

  // Lazy load AI explanations
  const toggleExplanation = async (idx: number, risk: RiskClause, llmAvailable: boolean) => {
    const isExpanded = !!expandedClauses[idx];
    setExpandedClauses(prev => ({ ...prev, [idx]: !isExpanded }));

    // If we're opening and don't have explanation yet
    if (!isExpanded && !explanations[idx]) {
      if (!llmAvailable) {
        setExplanations(prev => ({ 
          ...prev, 
          [idx]: risk.explanation || 'Template-based analysis suggests reviewing liability liability limits.' 
        }));
        return;
      }

      setLoadingExplanations(prev => ({ ...prev, [idx]: true }));
      try {
        const resp = await fetch('/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sentence: risk.original_sentence,
            risk_type: risk.risk_type,
            severity: risk.severity,
          })
        });
        const payload = await resp.json();
        
        setLoadingExplanations(prev => ({ ...prev, [idx]: false }));
        if (resp.ok && payload.explanation) {
          setExplanations(prev => ({ ...prev, [idx]: payload.explanation }));
        } else {
          setExplanations(prev => ({ ...prev, [idx]: payload.error || 'Failed to generate explanation.' }));
        }
      } catch (err: any) {
        setLoadingExplanations(prev => ({ ...prev, [idx]: false }));
        setExplanations(prev => ({ ...prev, [idx]: 'Unable to connect to explanation backend.' }));
      }
    }
  };

  // Interactive Custom SVG Doughnut Chart Calculation
  const renderSvgDoughnut = () => {
    if (!analysisResult || analysisResult.risks.length === 0) return null;
    
    // Group risk types
    const counts: Record<string, number> = {};
    analysisResult.risks.forEach(r => {
      counts[r.risk_type] = (counts[r.risk_type] || 0) + 1;
    });

    const categories = Object.keys(counts);
    const dataValues = Object.values(counts);
    const total = dataValues.reduce((a, b) => a + b, 0);

    const colors = ['#000000', '#333333', '#666666', '#999999', '#cccccc'];
    
    const radius = 50;
    const strokeWidth = 14;
    const circumference = 2 * Math.PI * radius;
    
    let accumulatedCircumference = 0;

    return (
      <div className="flex flex-col sm:flex-row items-center justify-center gap-8 my-6">
        {/* SVG Circle */}
        <div className="relative w-44 h-44">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
            {/* Background ring */}
            <circle
              cx="60"
              cy="60"
              r={radius}
              fill="transparent"
              stroke="#f3f4f6"
              strokeWidth={strokeWidth}
            />
            {categories.map((cat, idx) => {
              const val = counts[cat];
              const pct = val / total;
              const strokeLength = pct * circumference;
              const strokeOffset = circumference - strokeLength + accumulatedCircumference;
              accumulatedCircumference -= strokeLength;

              return (
                <circle
                  key={cat}
                  cx="60"
                  cy="60"
                  r={radius}
                  fill="transparent"
                  stroke={colors[idx % colors.length]}
                  strokeWidth={strokeWidth}
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeOffset}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out"
                />
              );
            })}
          </svg>
          {/* Middle text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-black tracking-tight">{total}</span>
            <span className="text-[10px] uppercase tracking-wider text-black/40 font-medium">Risk Clauses</span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-col gap-2.5 max-w-[220px]">
          {categories.map((cat, idx) => (
            <div key={cat} className="flex items-center gap-3">
              <span 
                className="w-3.5 h-3.5 rounded-full shrink-0 border border-black/10" 
                style={{ backgroundColor: colors[idx % colors.length] }}
              />
              <div className="flex justify-between items-center w-full gap-4">
                <span className="text-[14px] font-medium text-black/80 truncate leading-none">{cat}</span>
                <span className="text-[14px] font-black text-black leading-none">{counts[cat]}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="relative min-h-screen selection:bg-black/10 text-black bg-[#e5e5e5]">
      
      {/* ── NAVBAR (fixed, z-index: 10) ──────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center px-5 sm:px-8 py-4 sm:py-5 border-b border-black/5 backdrop-blur-md bg-[#e5e5e5]/50">
        
        {/* Logo (left) */}
        <div 
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} 
          className="flex items-center gap-3 cursor-pointer group select-none"
        >
          <span className="text-[21px] sm:text-[26px] tracking-tight text-black font-heading font-medium leading-none">
            LexGuard®
          </span>
          <span className="text-[25px] sm:text-[30px] text-black select-none tracking-tighter leading-none -mt-1 group-hover:rotate-45 transition-transform duration-300">
            ✳︎
          </span>
        </div>

        {/* Desktop CTA (right) */}
        <div>
          <button 
            onClick={() => scrollToSection(contactSectionRef)} 
            className="text-[23px] text-black underline underline-offset-4 hover:opacity-60 transition-opacity font-heading font-medium"
          >
            Get in touch
          </button>
        </div>
      </nav>

      {/* ── HERO SECTION (z-index: 1) ────────────────────────────────────── */}
      <section 
        className="relative h-screen w-full flex flex-col md:flex-row items-center justify-between px-5 sm:px-8 md:px-12 pt-24 overflow-hidden z-10 select-none animate-fadeIn"
        style={{ background: 'radial-gradient(circle at 80% 40%, #f3f3f3 0%, #e2e2e2 100%)' }}
      >
        
        {/* Left Side: Brand & Call to Actions */}
        <div className="max-w-xl flex flex-col justify-center h-full flex-1 z-10 pr-0 md:pr-6">
          {/* Section Indicator */}
          <div className="text-[12px] uppercase tracking-[3px] text-black/55 font-bold mb-4 font-heading">
            Legal Risk Diagnostics
          </div>

          {/* Heading */}
          <h1 className="text-4xl sm:text-6xl font-black text-black tracking-tight leading-none uppercase mb-6 font-heading">
            Automated Legal
            <br />
            Contract Assurance
          </h1>

          {/* Typewriter text */}
          <div 
            className="text-black/85 mb-8 font-light leading-[1.35] min-h-[72px] tracking-tight font-body"
            style={{ fontSize: 'clamp(17px, 3.5vw, 22px)' }}
          >
            {typedText}
            {!typewriterDone && (
              <span className="inline-block w-[2px] h-[1.1em] bg-black align-middle ml-[2px] animate-cursor-blink" />
            )}
          </div>

          {/* Action CTA Button redirecting to analyzer */}
          <div 
            className={`transition-all duration-700 ease-out transform ${
              showCTA ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'
            }`}
          >
            <button 
              onClick={() => scrollToSection(analyzerSectionRef)}
              className="inline-flex items-center justify-center bg-black text-white border border-black/10 rounded-full text-[14px] sm:text-[16px] px-7 py-3.5 cursor-pointer hover:bg-white hover:text-black hover:scale-105 active:scale-95 transition-all duration-200 shadow-xl font-bold gap-2.5"
            >
              <span>Analyze Document</span>
              <span className="text-[18px] leading-none">✳︎</span>
            </button>
          </div>
        </div>

        {/* Right Side: Reactive, Animated Scales of Justice */}
        <div className="flex-1 w-full max-w-[320px] sm:max-w-[420px] md:max-w-[480px] h-full flex items-center justify-center z-10 py-6 md:py-0">
          <svg 
            onMouseMove={handleScaleMouseMove}
            onMouseLeave={handleScaleMouseLeave}
            viewBox="0 0 300 300" 
            className="w-full h-auto text-black drop-shadow-2xl cursor-crosshair select-none"
          >
            {/* Stand Base */}
            <path d="M70 270 L230 270" stroke="currentColor" strokeWidth="6" strokeLinecap="round" />
            <path d="M110 270 L110 260 L190 260 L190 270 Z" fill="currentColor" />
            
            {/* Vertical Stand Post */}
            <path d="M150 260 L150 78" stroke="currentColor" strokeWidth="6" strokeLinecap="round" />
            
            {/* Stand Pivot Point Circle */}
            <circle cx="150" cy="78" r="8" fill="currentColor" />
            <path d="M150 78 L150 63" stroke="currentColor" strokeWidth="3" />
            
            {/* Horizontal Tilting Balance Beam + Pans (Nested for correct translation) */}
            <g style={{
              transform: `rotate(${tilt}deg)`,
              transformOrigin: '150px 78px'
            }}>
              {/* Beam line */}
              <path d="M50 78 L250 78" stroke="currentColor" strokeWidth="5" strokeLinecap="round" />
              
              {/* Pivot markers */}
              <circle cx="50" cy="78" r="4.5" fill="currentColor" />
              <circle cx="250" cy="78" r="4.5" fill="currentColor" />
              
              {/* Centre pointer */}
              <path d="M150 78 L150 94" stroke="currentColor" strokeWidth="3.5" />

              {/* Left Balance Pan (Nested inside parent group, counter-rotated around pivot point 50,78) */}
              <g style={{
                transform: `rotate(${-tilt}deg)`,
                transformOrigin: '50px 78px'
              }}>
                {/* Suspension Chains */}
                <path d="M50 78 L25 180 M50 78 L75 180" stroke="currentColor" strokeWidth="1.5" opacity="0.65" />
                {/* Scale Pan Plate */}
                <path d="M20 180 L80 180" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                {/* Subtle fill */}
                <path d="M20 180 Q50 202 80 180" fill="currentColor" opacity="0.12" />
              </g>
              
              {/* Right Balance Pan (Nested inside parent group, counter-rotated around pivot point 250,78) */}
              <g style={{
                transform: `rotate(${-tilt}deg)`,
                transformOrigin: '250px 78px'
              }}>
                {/* Suspension Chains */}
                <path d="M250 78 L225 180 M250 78 L275 180" stroke="currentColor" strokeWidth="1.5" opacity="0.65" />
                {/* Scale Pan Plate */}
                <path d="M220 180 L280 180" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                {/* Subtle fill */}
                <path d="M220 180 Q250 202 280 180" fill="currentColor" opacity="0.12" />
              </g>
            </g>
          </svg>
        </div>

      </section>

      {/* ── ANALYZER CONSOLE SECTION (under the fold) ────────────────────── */}
      <div 
        ref={analyzerSectionRef}
        id="analyzer-section" 
        className="relative z-10 w-full min-h-screen bg-black/95 text-white/90 backdrop-blur-xl border-t border-white/10 px-5 sm:px-8 md:px-12 py-20 flex flex-col items-center"
      >
        <div className="max-w-4xl w-full">
          
          {/* Main Title */}
          <div className="text-center mb-12">
            <span className="text-[12px] uppercase tracking-[3px] text-white/50 font-bold border border-white/20 px-3 py-1 rounded-full bg-white/5 font-heading">
              Confidence-Calibrated Inference Cascade
            </span>
            <h2 className="text-3xl sm:text-5xl font-black mt-4 tracking-tight text-white uppercase font-heading">
              Contract Risk Detector
            </h2>
            <p className="text-white/60 max-w-xl mx-auto mt-3 text-sm sm:text-[16px] leading-relaxed font-light font-body">
              Submit contract PDFs or paste clauses. LexGuard classifies risk levels utilizing Legal-BERT and details anomalies.
            </p>
          </div>

          {/* Form Card */}
          <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-6 sm:p-8 backdrop-blur-md shadow-2xl">
            <form onSubmit={handleAnalyze} className="space-y-6">
              
              {/* File upload zone */}
              <div className="flex flex-col space-y-2">
                <label className="text-[11px] font-bold text-white/60 uppercase tracking-widest font-heading">
                  Upload Contract File
                </label>
                <div className="relative group border-2 border-dashed border-white/15 hover:border-white/40 transition-colors duration-200 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer bg-white/[0.01]">
                  <input
                    type="file"
                    accept=".pdf,.txt"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setFile(e.target.files[0]);
                        setTextInput(''); // Clear pasted text if file uploaded
                      }
                    }}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <svg className="w-10 h-10 text-white/30 group-hover:text-white/60 transition-colors duration-200 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <span className="text-sm font-medium text-white/80">
                    {file ? file.name : 'Choose a file or drag it here'}
                  </span>
                  <span className="text-[11px] text-white/40 mt-1">
                    Accepts PDF or TXT up to 10MB
                  </span>
                </div>
              </div>

              {/* OR divider */}
              <div className="flex items-center justify-center gap-4 my-2 select-none">
                <div className="h-[1px] bg-white/10 flex-1" />
                <span className="text-[11px] font-bold tracking-widest text-white/40 uppercase">OR</span>
                <div className="h-[1px] bg-white/10 flex-1" />
              </div>

              {/* Text Area */}
              <div className="flex flex-col space-y-2">
                <label htmlFor="pasted-text" className="text-[11px] font-bold text-white/60 uppercase tracking-widest font-heading">
                  Paste Contract Clauses
                </label>
                <textarea
                  id="pasted-text"
                  rows={6}
                  placeholder="Paste specific terms or whole agreements here to analyze..."
                  value={textInput}
                  onChange={(e) => {
                    setTextInput(e.target.value);
                    setFile(null); // Clear file if text entered
                  }}
                  className="bg-white/[0.04] focus:bg-white/[0.07] border border-white/10 focus:border-white/30 rounded-xl p-4 text-sm text-white placeholder-white/20 outline-none transition-all duration-200 resize-y"
                />
              </div>

              {/* Submit button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex items-center justify-center py-3.5 bg-white text-black font-black text-[15px] uppercase tracking-wider rounded-xl cursor-pointer hover:bg-white/90 active:scale-[0.99] transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-xl"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <svg className="animate-spin h-5 w-5 text-black" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Running CCIC Pipeline...</span>
                  </div>
                ) : (
                  <span>⚡ Analyze with LexGuard</span>
                )}
              </button>
            </form>
          </div>

          {/* Loading status details */}
          {isLoading && (
            <div className="text-center mt-6 text-white/40 text-xs animate-pulse font-light font-body">
              Legal-BERT is computing token distributions and evaluating confidence cascade...
            </div>
          )}

          {/* ── RESULTS DASHBOARD ────────────────────────────────────────── */}
          {analysisResult && (
            <div id="results-dashboard" className="mt-16 space-y-10 scroll-mt-24">
              
              {/* Summary details card */}
              <div className="bg-white text-black rounded-2xl p-6 sm:p-8 border border-black/10 shadow-2xl transition-all">
                <h3 className="text-xl sm:text-2xl font-black tracking-tight border-b border-black/10 pb-4 mb-6 font-heading">
                  📊 Analysis Summary
                </h3>

                {/* Stat numbers */}
                <div className="grid grid-cols-3 gap-2 sm:gap-6 text-center">
                  <div className="flex flex-col justify-center">
                    <span className="text-3xl sm:text-5xl font-black tracking-tight">
                      {analysisResult.summary.total_sentences}
                    </span>
                    <span className="text-[10px] font-bold text-black/40 uppercase tracking-wider mt-1 font-heading">
                      Total Sentences
                    </span>
                  </div>
                  <div className="h-10 w-[1px] bg-black/10 self-center justify-self-center" />
                  <div className="flex flex-col justify-center">
                    <span className="text-3xl sm:text-5xl font-black tracking-tight text-red-500">
                      {analysisResult.summary.risky_count}
                    </span>
                    <span className="text-[10px] font-bold text-black/40 uppercase tracking-wider mt-1 font-heading">
                      Risky Clauses
                    </span>
                  </div>
                  <div className="h-10 w-[1px] bg-black/10 self-center justify-self-center" />
                  <div className="flex flex-col justify-center">
                    <span className="text-3xl sm:text-5xl font-black tracking-tight">
                      {analysisResult.summary.risk_percentage}%
                    </span>
                    <span className="text-[10px] font-bold text-black/40 uppercase tracking-wider mt-1 font-heading">
                      Risk Level
                    </span>
                  </div>
                </div>

                <div className="h-[1px] bg-black/10 my-6" />

                {/* Severity Pills */}
                <div className="flex flex-wrap gap-2 items-center justify-center mb-6">
                  <span className="text-[11px] font-bold text-black/40 uppercase tracking-wider mr-2 font-heading">Severity:</span>
                  <span className="inline-flex items-center text-xs font-black bg-red-50 text-red-600 border border-red-200 px-3 py-1 rounded-full gap-1.5 font-heading">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> High {analysisResult.summary.severity_distribution.High || 0}
                  </span>
                  <span className="inline-flex items-center text-xs font-black bg-amber-50 text-amber-600 border border-amber-200 px-3 py-1 rounded-full gap-1.5 font-heading">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> Medium {analysisResult.summary.severity_distribution.Medium || 0}
                  </span>
                  <span className="inline-flex items-center text-xs font-black bg-emerald-50 text-emerald-600 border border-emerald-200 px-3 py-1 rounded-full gap-1.5 font-heading">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Low {analysisResult.summary.severity_distribution.Low || 0}
                  </span>
                </div>

                {/* Doughnut Chart */}
                {renderSvgDoughnut()}

              </div>

              {/* CCIC Config info board */}
              <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4 sm:p-5 flex flex-wrap gap-4 items-center justify-between text-xs text-white/70">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="font-bold text-white uppercase tracking-wider font-heading">CCIC Configuration</span>
                  <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 font-mono text-[10px]">
                    Mode: {analysisResult.ccic.mode}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 font-mono text-[10px]">
                    τ_t: {analysisResult.ccic.transformer_threshold}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 font-mono text-[10px]">
                    τ_m: {analysisResult.ccic.ml_threshold}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 font-mono text-[10px]">
                    T: {analysisResult.ccic.temperature}
                  </span>
                </div>
                
                <div className="flex items-center gap-4 text-[11px] font-mono border-t border-white/5 sm:border-t-0 pt-2 sm:pt-0">
                  <span className="title" title="Classified by transformer model">⚡ Trans: {analysisResult.ccic.detector_distribution.transformer || 0}</span>
                  <span className="title" title="Classified by ML classifier">🤖 ML: {analysisResult.ccic.detector_distribution.ml || 0}</span>
                  <span className="title" title="Classified by heuristic keywords">🔑 Key: {analysisResult.ccic.detector_distribution.keyword || 0}</span>
                </div>
              </div>

              {/* Detected risks list */}
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-2 border-b border-white/10">
                  <h4 className="text-lg font-black uppercase tracking-wider text-white font-heading">
                    🔍 Detected Anomalies
                  </h4>
                  <span className="text-xs bg-white/10 px-3 py-1 rounded-full text-white/80 font-bold border border-white/5 font-heading">
                    {analysisResult.risks.length} Risk Clauses
                  </span>
                </div>

                {analysisResult.risks.length === 0 ? (
                  <div className="text-center py-16 bg-white/[0.02] border border-white/10 rounded-xl text-white/40">
                    <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="font-bold">No risky clauses detected</p>
                    <p className="text-xs mt-1 font-light">This contract appears to carry minimal standard risk variables.</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {analysisResult.risks.map((risk, idx) => {
                      const isExpanded = !!expandedClauses[idx];
                      const explanation = explanations[idx];
                      const isExplLoading = !!loadingExplanations[idx];

                      let sevColor = '';
                      let sevBg = '';
                      let borderStyle = '';
                      
                      if (risk.severity === 'High') {
                        sevColor = 'text-red-500 border-red-500/20';
                        sevBg = 'bg-red-500';
                        borderStyle = 'border-red-500/30 hover:border-red-500/50';
                      } else if (risk.severity === 'Medium') {
                        sevColor = 'text-amber-500 border-amber-500/20';
                        sevBg = 'bg-amber-500';
                        borderStyle = 'border-amber-500/30 hover:border-amber-500/50';
                      } else {
                        sevColor = 'text-emerald-500 border-emerald-500/20';
                        sevBg = 'bg-emerald-500';
                        borderStyle = 'border-emerald-500/20 hover:border-emerald-500/40';
                      }

                      return (
                        <div
                          key={idx}
                          onClick={() => toggleExplanation(idx, risk, analysisResult.llm_available)}
                          className={`relative overflow-hidden bg-white/[0.02] hover:bg-white/[0.04] border ${borderStyle} rounded-xl p-5 sm:p-6 transition-all duration-200 cursor-pointer shadow-lg`}
                        >
                          {/* Severity left colored strip */}
                          <div className={`absolute top-0 left-0 bottom-0 w-1 ${sevBg}`} />

                          {/* Card header */}
                          <div className="flex justify-between items-center flex-wrap gap-2 mb-3">
                            <span className="text-[12px] font-black uppercase tracking-wider text-white font-heading">
                              {risk.risk_type} Risk
                            </span>
                            <span className={`text-[10px] font-bold uppercase tracking-wider border px-2 py-0.5 rounded-full font-heading ${sevColor}`}>
                              {risk.severity} Severity
                            </span>
                          </div>

                          {/* Clause Text */}
                          <p className="text-white/80 text-sm font-light italic leading-relaxed mb-4 font-body">
                            "{risk.original_sentence}"
                          </p>

                          {/* Score bar */}
                          {typeof risk.severity_score === 'number' && (
                            <div className="flex items-center gap-3 mb-4 max-w-md">
                              <div className="h-1.5 bg-white/5 rounded-full flex-1 overflow-hidden">
                                <div 
                                  className={`h-full rounded-full transition-all duration-1000 ${sevBg}`}
                                  style={{ width: `${Math.round(risk.severity_score * 100)}%` }}
                                />
                              </div>
                              <span className="text-[10px] font-mono text-white/50 shrink-0">
                                score: {risk.severity_score.toFixed(3)}
                              </span>
                            </div>
                          )}

                          {/* Meta tags line */}
                          <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-white/40 border-t border-white/5 pt-3">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className={`px-2 py-0.5 rounded font-mono text-[9px] ${
                                risk.detector === 'transformer' 
                                  ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' 
                                  : risk.detector === 'ml'
                                    ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                              }`}>
                                {risk.detector === 'transformer' ? '⚡ Transformer' : risk.detector === 'ml' ? '🤖 ML' : '🔑 Keyword'}
                              </span>
                              {typeof risk.confidence === 'number' && (
                                <span className="px-2 py-0.5 rounded bg-white/5 border border-white/5 font-mono text-[9px]">
                                  conf: {risk.confidence.toFixed(4)}
                                </span>
                              )}
                            </div>
                            <span className="text-white hover:text-white/80 transition-colors flex items-center gap-1 font-bold">
                              {isExpanded ? '▴ Hide' : '▾ Explain'}
                            </span>
                          </div>

                          {/* Collapsible explanation drawer */}
                          {isExpanded && (
                            <div 
                              className="mt-4 p-4 rounded-lg bg-white/[0.03] border border-white/10 text-xs sm:text-sm text-white/70 leading-relaxed space-y-1.5 animate-fadeIn"
                              onClick={(e) => e.stopPropagation()} // Stop click bubbling up
                            >
                              <strong className="text-white font-black block uppercase tracking-wider text-[10px] text-white/60 font-heading">
                                AI Explanation
                              </strong>
                              {isExplLoading ? (
                                <div className="flex items-center gap-2 text-white/50 py-1 font-light animate-pulse font-body">
                                  <svg className="animate-spin h-4 w-4 text-white/60" viewBox="0 0 24 24" fill="none">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                  </svg>
                                  <span>Generating explanation clause...</span>
                                </div>
                              ) : (
                                <p className="font-light font-body">{explanation}</p>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

            </div>
          )}

        </div>
      </div>

      {/* ── FOOTER & CONTACT SECTION (Get in Touch) ────────────────────── */}
      <footer 
        ref={contactSectionRef}
        id="contact" 
        className="relative z-10 w-full bg-black text-white/50 py-16 px-5 sm:px-8 md:px-12 border-t border-white/10 text-center"
      >
        <div className="max-w-4xl mx-auto space-y-8">
          
          <div className="space-y-3">
            <h3 className="text-2xl sm:text-4xl font-black text-white uppercase tracking-tight font-heading">
              LexGuard Systems
            </h3>
            <p className="max-w-md mx-auto text-xs sm:text-sm font-light leading-relaxed font-body">
              LexGuard is an academic preview engineered by LexGuard Legal Informatics Lab. Verify critical clauses with official counsel before execution.
            </p>
          </div>

          {/* User Profile Details Fetched dynamically from GitHub */}
          {githubProfile && (
            <div className="max-w-sm mx-auto bg-white/[0.03] border border-white/10 rounded-xl p-4 flex items-center gap-4 text-left shadow-lg backdrop-blur-sm animate-fadeIn">
              <a 
                href={githubProfile.html_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="shrink-0 rounded-full overflow-hidden border border-white/20 hover:border-white/60 transition-colors"
              >
                <img 
                  src={githubProfile.avatar_url} 
                  alt={githubProfile.name}
                  className="w-12 h-12 object-cover"
                />
              </a>
              <div className="flex-1 min-w-0">
                <a 
                  href={githubProfile.html_url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-white hover:underline font-bold text-sm truncate block font-heading"
                >
                  {githubProfile.name}
                </a>
                <span className="text-[11px] text-white/50 block font-mono">
                  @{githubProfile.login}
                </span>
                <p className="text-[11px] text-white/70 mt-1 line-clamp-2 leading-snug font-body font-light">
                  {githubProfile.bio}
                </p>
              </div>
              <div className="shrink-0 text-right border-l border-white/10 pl-3">
                <div className="text-[11px] font-mono">
                  <span className="text-white font-bold">{githubProfile.public_repos}</span>
                  <span className="text-white/40 block text-[9px] uppercase tracking-wider">Repos</span>
                </div>
                <div className="text-[11px] font-mono mt-1">
                  <span className="text-white font-bold">{githubProfile.followers}</span>
                  <span className="text-white/40 block text-[9px] uppercase tracking-wider">Followers</span>
                </div>
              </div>
            </div>
          )}

          {/* Contact Details Grid / List */}
          <div className="flex flex-col space-y-4">
            <h4 className="text-[11px] font-bold text-white/60 uppercase tracking-widest font-heading">
              Get in Touch
            </h4>
            <div className="flex flex-wrap justify-center gap-x-8 gap-y-3 text-xs sm:text-sm font-body">
              <a 
                href="mailto:hemanthkumar.amarthi7@gmail.com" 
                className="text-white hover:text-white/70 transition-colors flex items-center gap-2.5 font-light"
              >
                <svg className="w-4 h-4 shrink-0 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span>hemanthkumar.amarthi7@gmail.com</span>
              </a>
              <a 
                href="tel:8919684910" 
                className="text-white hover:text-white/70 transition-colors flex items-center gap-2.5 font-light"
              >
                <svg className="w-4 h-4 shrink-0 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.94.725l.548 2.2a1 1 0 01-.321.988l-1.305.98a10.582 10.582 0 004.872 4.872l.98-1.305a1 1 0 01.988-.321l2.2.548a1 1 0 01.725.94V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
                <span>+91 8919684910</span>
              </a>
              <a 
                href="https://www.linkedin.com/in/hemanth-kumar-amarthi/" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-white hover:text-white/70 transition-colors flex items-center gap-2.5 font-light"
              >
                <svg className="w-4 h-4 shrink-0 text-white/60" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.779-1.75-1.75s.784-1.75 1.75-1.75 1.75.779 1.75 1.75-.784 1.75-1.75 1.75zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                </svg>
                <span>LinkedIn</span>
              </a>
              <a 
                href="https://hemanth-portfolio-omega-gray.vercel.app/" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-white hover:text-white/70 transition-colors flex items-center gap-2.5 font-light"
              >
                <svg className="w-4 h-4 shrink-0 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                </svg>
                <span>Portfolio</span>
              </a>
            </div>
          </div>

          <div className="flex justify-center gap-8 text-xs uppercase tracking-wider font-bold">
            <a 
              href="https://huggingface.co/spaces/Hemanth021/legal-risk-detector" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:text-white/60 transition-colors"
            >
              HuggingFace Space
            </a>
            <span>·</span>
            <a 
              href="https://github.com/HEMANTH-A-7/Legal-Risk-Detector" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:text-white/60 transition-colors"
            >
              GitHub Project
            </a>
          </div>

          <div className="h-[1px] bg-white/10 w-24 mx-auto" />

          <p className="text-[10px] tracking-widest uppercase">
            © {new Date().getFullYear()} LexGuard. All Rights Reserved.
          </p>
        </div>
      </footer>

    </div>
  );
};

export default App;
