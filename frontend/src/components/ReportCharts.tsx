import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { InsightsReport } from '../types/api';

interface ReportChartsProps {
  report: InsightsReport;
}

const SENTIMENT_COLORS: Record<string, string> = {
  Positive: '#22c55e',
  Negative: '#ef4444',
  Mixed: '#f59e0b',
  Neutral: '#9ca3af',
  Unknown: '#d1d5db',
};

const AXIS_STYLE = { fontSize: 12, fill: '#6b7280' };

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-gray-100 p-4">
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-wide text-gray-500">{title}</h4>
      {children}
    </div>
  );
}

export function ReportCharts({ report }: ReportChartsProps) {
  const ratingData = ['1', '2', '3', '4', '5'].map((stars) => ({
    stars: `${stars}★`,
    count: report.rating_distribution[stars] ?? 0,
  }));

  const sentimentData = Object.entries(report.sentiment_breakdown)
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => ({ name: label, value: count }));

  const sourceData = Object.entries(report.source_breakdown)
    .sort((a, b) => b[1] - a[1])
    .map(([source, count]) => ({ source, count }));

  const praisedData = report.top_praised
    .slice(0, 8)
    .map(([entity, count]) => ({ entity, count }))
    .reverse();

  const complainedData = report.top_complained
    .slice(0, 8)
    .map(([entity, count]) => ({ entity, count }))
    .reverse();

  const hasAnyData =
    ratingData.some((d) => d.count > 0) || sentimentData.length > 0 || sourceData.length > 0;

  if (!hasAnyData) return null;

  return (
    <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
      <ChartCard title="Rating Distribution">
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={ratingData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <XAxis dataKey="stars" tick={AXIS_STYLE} axisLine={false} tickLine={false} />
            <YAxis allowDecimals={false} tick={AXIS_STYLE} axisLine={false} tickLine={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#E85D3C" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Sentiment Breakdown">
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie
              data={sentimentData}
              dataKey="value"
              nameKey="name"
              innerRadius={40}
              outerRadius={70}
              paddingAngle={2}
            >
              {sentimentData.map((entry) => (
                <Cell key={entry.name} fill={SENTIMENT_COLORS[entry.name] ?? '#9ca3af'} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
      </ChartCard>

      {sourceData.length > 0 && (
        <ChartCard title="Reviews by Platform">
          <ResponsiveContainer width="100%" height={Math.max(120, sourceData.length * 36)}>
            <BarChart
              data={sourceData}
              layout="vertical"
              margin={{ top: 4, right: 16, left: 8, bottom: 0 }}
            >
              <XAxis type="number" allowDecimals={false} tick={AXIS_STYLE} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="source"
                tick={AXIS_STYLE}
                axisLine={false}
                tickLine={false}
                width={80}
              />
              <Tooltip />
              <Bar dataKey="count" fill="#E85D3C" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {(praisedData.length > 0 || complainedData.length > 0) && (
        <ChartCard title="Top Praised vs. Complained">
          <div className="grid grid-cols-1 gap-3">
            {praisedData.length > 0 && (
              <ResponsiveContainer width="100%" height={Math.max(100, praisedData.length * 28)}>
                <BarChart
                  data={praisedData}
                  layout="vertical"
                  margin={{ top: 0, right: 16, left: 8, bottom: 0 }}
                >
                  <XAxis type="number" allowDecimals={false} hide />
                  <YAxis
                    type="category"
                    dataKey="entity"
                    tick={AXIS_STYLE}
                    axisLine={false}
                    tickLine={false}
                    width={100}
                  />
                  <Tooltip />
                  <Bar dataKey="count" fill="#22c55e" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
            {complainedData.length > 0 && (
              <ResponsiveContainer width="100%" height={Math.max(100, complainedData.length * 28)}>
                <BarChart
                  data={complainedData}
                  layout="vertical"
                  margin={{ top: 0, right: 16, left: 8, bottom: 0 }}
                >
                  <XAxis type="number" allowDecimals={false} hide />
                  <YAxis
                    type="category"
                    dataKey="entity"
                    tick={AXIS_STYLE}
                    axisLine={false}
                    tickLine={false}
                    width={100}
                  />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </ChartCard>
      )}
    </div>
  );
}
