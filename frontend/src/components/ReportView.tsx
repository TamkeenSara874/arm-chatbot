import { Download, Loader2, X } from 'lucide-react';
import { useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useReport } from '../hooks/useChat';
import { useChatStore } from '../store/chatStore';
import { ReportCharts } from './ReportCharts';

interface ReportViewProps {
  onClose: () => void;
}

export function ReportView({ onClose }: ReportViewProps) {
  const { restaurantId, sessionId } = useChatStore();
  const { mutate: generateReport, data, isPending, error, reset } = useReport();
  const [isExporting, setIsExporting] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  function handleGenerate() {
    if (!restaurantId || !sessionId) return;
    generateReport({
      session_id: sessionId,
      restaurant_id: restaurantId,
      message: 'Generate a full insights report for this restaurant.',
    });
  }

  async function handleDownloadPdf() {
    if (!data?.report.markdown || !contentRef.current) return;
    setIsExporting(true);
    let clone: HTMLElement | null = null;
    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
        import('html2canvas'),
        import('jspdf'),
      ]);

      // Clone into a detached, full-height div so html2canvas captures everything
      clone = contentRef.current.cloneNode(true) as HTMLElement;
      clone.style.cssText =
        'position:absolute;top:-99999px;left:-99999px;width:750px;padding:32px;background:#fff;';
      document.body.appendChild(clone);

      const canvas = await html2canvas(clone, {
        scale: 2,
        backgroundColor: '#ffffff',
        useCORS: true,
      });

      const pdf = new jsPDF({ orientation: 'p', unit: 'pt', format: 'a4' });
      const pdfW = pdf.internal.pageSize.getWidth();
      const pdfH = pdf.internal.pageSize.getHeight();
      const ratio = pdfW / canvas.width;
      const totalH = canvas.height * ratio;
      const imgData = canvas.toDataURL('image/jpeg', 0.92);

      let pageY = 0;
      let page = 0;
      while (pageY < totalH) {
        if (page > 0) pdf.addPage();
        pdf.addImage(imgData, 'JPEG', 0, -pageY, pdfW, totalH);
        pageY += pdfH;
        page++;
      }

      pdf.save(`review-insights-restaurant-${restaurantId}.pdf`);
    } finally {
      if (clone?.parentNode) {
        document.body.removeChild(clone);
      }
      setIsExporting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
      <div className="mx-4 flex h-[90vh] w-full max-w-3xl flex-col rounded-xl bg-white shadow-xl animate-fade-in">
        <div className="flex shrink-0 items-center justify-between border-b border-gray-100 px-5 py-4">
          <h2 className="text-sm font-semibold text-gray-800">Insights Report</h2>
          <div className="flex items-center gap-2">
            {data?.report.markdown && (
              <button
                onClick={handleDownloadPdf}
                disabled={isExporting}
                className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-50 disabled:opacity-50"
              >
                {isExporting ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Download size={13} />
                )}
                {isExporting ? 'Exporting…' : 'Download PDF'}
              </button>
            )}
            <button
              onClick={() => { reset(); onClose(); }}
              className="rounded-md p-1 text-gray-400 transition hover:bg-gray-100 hover:text-gray-600"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
          {!data && !isPending && !error && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-aio-50">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="mb-1 text-sm font-semibold text-gray-800">Generate Insights Report</h3>
              <p className="mb-5 max-w-xs text-xs text-gray-400">
                Get a full breakdown of ratings, sentiments, top praised items, and complaints
                across all reviews for this restaurant.
              </p>
              <button
                onClick={handleGenerate}
                disabled={!sessionId}
                className="rounded-lg bg-aio-500 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-aio-600 disabled:opacity-50"
              >
                Generate Report
              </button>
            </div>
          )}

          {isPending && (
            <div className="flex flex-col items-center justify-center py-16 gap-3 text-gray-400">
              <Loader2 size={28} className="animate-spin text-aio-400" />
              <p className="text-sm">Analysing your reviews…</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <p className="mb-3 text-sm text-red-500">Failed to generate report. Please try again.</p>
              <button
                onClick={handleGenerate}
                className="rounded-lg bg-aio-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-aio-600"
              >
                Retry
              </button>
            </div>
          )}

          {data?.report.markdown && (
            <div ref={contentRef}>
              <ReportCharts report={data.report} />
              <div
                className="prose prose-sm prose-gray max-w-none
                  prose-headings:font-semibold prose-headings:text-gray-800
                  prose-h1:text-xl prose-h1:border-b prose-h1:border-gray-200 prose-h1:pb-2
                  prose-h2:text-base prose-h2:mt-6
                  prose-h3:text-sm
                  prose-p:text-gray-700 prose-p:leading-relaxed
                  prose-li:text-gray-700
                  prose-table:text-sm
                  prose-th:bg-gray-50 prose-th:text-gray-600 prose-th:font-semibold prose-th:text-xs
                  prose-td:text-gray-700
                  prose-strong:text-gray-900
                  prose-hr:border-gray-200"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {data.report.markdown}
                </ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
