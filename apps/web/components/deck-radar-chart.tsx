import { useMemo } from 'react';

export type RadarCharacteristic = {
  label: string;
  score: number;
  description: string;
  id?: string;
};

const GRID_STEPS = [2, 4, 6, 8, 10];
const CHART_SIZE = 320;
const CENTER = CHART_SIZE / 2;
const RADIUS = CENTER - 28;
const APPROX_CHAR_WIDTH = 6.2;

function polarToCartesian(angle: number, radius: number) {
  const x = CENTER + radius * Math.cos(angle);
  const y = CENTER + radius * Math.sin(angle);
  return { x, y };
}

function wrapLabel(label: string, maxCharsPerLine = 12): string[] {
  const words = label.split(' ');
  const lines: string[] = [];
  let current = '';

  words.forEach((word) => {
    const next = current ? `${current} ${word}` : word;
    if (next.length > maxCharsPerLine && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  });

  if (current) {
    lines.push(current);
  }

  return lines;
}

function estimateTextWidth(line: string): number {
  return line.length * APPROX_CHAR_WIDTH;
}

function formatValue(value: number): string {
  return Number.isInteger(value) ? value.toString() : value.toFixed(1);
}

export function DeckRadarChart({
  characteristics,
  maxValue = 10,
}: {
  characteristics: RadarCharacteristic[];
  maxValue?: number;
}) {
  const axisCount = characteristics.length;
  const angleStep = (Math.PI * 2) / axisCount;

  const points = useMemo(() => {
    return characteristics.map((characteristic, index) => {
      const angle = angleStep * index - Math.PI / 2; // start at top
      const ratio = Math.max(0, Math.min(1, characteristic.score / maxValue));
      return polarToCartesian(angle, RADIUS * ratio);
    });
  }, [angleStep, characteristics, maxValue]);

  const polygonPoints = points.map((point) => `${point.x},${point.y}`).join(' ');

  const gridPolygons = GRID_STEPS.map((step) => {
    const ratio = step / maxValue;
    const pathPoints = characteristics
      .map((_, index) => {
        const angle = angleStep * index - Math.PI / 2;
        const { x, y } = polarToCartesian(angle, RADIUS * ratio);
        return `${x},${y}`;
      })
      .join(' ');

    return { step, pathPoints };
  });

  return (
    <div className="flex flex-col gap-6">
      <div className="relative mx-auto" style={{ width: CHART_SIZE, height: CHART_SIZE }}>
        <svg width={CHART_SIZE} height={CHART_SIZE} viewBox={`0 0 ${CHART_SIZE} ${CHART_SIZE}`}>
          <defs>
            <radialGradient id="radar-fill" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(96, 165, 250, 0.35)" />
              <stop offset="100%" stopColor="rgba(37, 99, 235, 0.25)" />
            </radialGradient>
          </defs>

          {gridPolygons.map(({ step, pathPoints }) => (
            <polygon
              key={`grid-${step}`}
              points={pathPoints}
              fill="none"
              stroke="rgba(148, 163, 184, 0.25)"
              strokeWidth={1}
            />
          ))}

          {[...Array(axisCount)].map((_, index) => {
            const angle = angleStep * index - Math.PI / 2;
            const { x, y } = polarToCartesian(angle, RADIUS);
            return (
              <line
                key={`axis-${index}`}
                x1={CENTER}
                y1={CENTER}
                x2={x}
                y2={y}
                stroke="rgba(148, 163, 184, 0.35)"
                strokeWidth={1}
              />
            );
          })}

          <polygon points={polygonPoints} fill="url(#radar-fill)" stroke="rgba(59, 130, 246, 0.6)" strokeWidth={2} />

          {points.map(({ x, y }, index) => (
            <circle key={`point-${index}`} cx={x} cy={y} r={4.5} fill="#60a5fa" stroke="#1d4ed8" strokeWidth={1.5} />
          ))}

          {[...Array(axisCount)].map((_, index) => {
            const angle = angleStep * index - Math.PI / 2;
            const labelDistance = RADIUS + 40;
            const { x, y } = polarToCartesian(angle, labelDistance);
            const characteristic = characteristics[index];

            const isLeftSide = Math.cos(angle) < -0.2;
            const isRightSide = Math.cos(angle) > 0.2;

            let anchor: 'middle' | 'start' | 'end' = 'middle';
            if (isLeftSide) anchor = 'end';
            if (isRightSide) anchor = 'start';

            const labelLines = wrapLabel(characteristic.label);
            const lineHeight = 12;
            const totalHeight = lineHeight * (labelLines.length - 1);
            const verticalPadding = 14;
            const horizontalPadding = 18;
            const longestLine = Math.max(...labelLines.map((line) => line.length));
            const approxLineWidth = longestLine * APPROX_CHAR_WIDTH;

            let clampedX = x;
            if (anchor === 'start') {
              clampedX = Math.min(x, CHART_SIZE - horizontalPadding - approxLineWidth);
            } else if (anchor === 'end') {
              clampedX = Math.max(x, horizontalPadding + approxLineWidth);
            } else {
              clampedX = Math.min(
                Math.max(x, horizontalPadding + approxLineWidth / 2),
                CHART_SIZE - horizontalPadding - approxLineWidth / 2,
              );
            }

            let startY = y - totalHeight / 2;
            if (startY < verticalPadding) {
              startY = verticalPadding;
            } else if (startY + totalHeight > CHART_SIZE - verticalPadding) {
              startY = CHART_SIZE - verticalPadding - totalHeight;
            }

            return (
              <text
                key={`label-${index}`}
                x={clampedX}
                y={startY}
                fill="rgb(226 232 240)"
                fontSize={11}
                textAnchor={anchor}
              >
                {labelLines.map((line, lineIndex) => (
                  <tspan key={lineIndex} x={clampedX} y={startY + lineIndex * lineHeight}>
                    {line}
                  </tspan>
                ))}
              </text>
            );
          })}

          {GRID_STEPS.map((step) => (
            <text
              key={`legend-${step}`}
              x={CENTER + 6}
              y={CENTER - (RADIUS * step) / maxValue}
              fill="rgba(148, 163, 184, 0.5)"
              fontSize={11}
            >
              {step}
            </text>
          ))}
        </svg>
      </div>

      <div className="grid gap-3 text-sm text-slate-200 md:grid-cols-2">
        {characteristics.map((characteristic) => (
          <div
            key={characteristic.id ?? characteristic.label}
            className="space-y-1 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-3"
          >
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-100">{characteristic.label}</span>
              <span className="text-primary-300">{formatValue(characteristic.score)}</span>
            </div>
            <p className="text-xs leading-relaxed text-slate-400">{characteristic.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
