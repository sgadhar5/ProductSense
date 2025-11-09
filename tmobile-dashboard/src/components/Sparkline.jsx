export default function Sparkline({ data = [], width = 140, height = 36 }) {
    const vals = data.filter(v => typeof v === "number");
    if (!vals.length) return <svg width={width} height={height} />;
  
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const norm = v => {
      if (max === min) return height / 2;
      const t = (v - min) / (max - min);
      return height - t * (height - 4) - 2;
    };
  
    const step = (width - 6) / (vals.length - 1);
    let d = "";
    vals.forEach((v, i) => {
      const x = 3 + i * step;
      const y = norm(v);
      d += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });
  
    return (
      <svg width={width} height={height} aria-label="sparkline">
        <path d={d} fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    );
  }
  