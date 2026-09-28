import { useState } from "react";
import DocumentWorkspace from "./components/DocumentWorkspace";
import type { DiffView } from "./components/DraftDiff";
import ReviewInspector from "./components/ReviewInspector";
import ReviewSidebar from "./components/ReviewSidebar";
import SettingsPage from "./components/SettingsPage";
import type { SetupPage } from "./SetupWizard";
import { useReviewWorkspace } from "./hooks/useReviewWorkspace";
import { useInspectorResize } from "./hooks/useInspectorResize";
import { decisionLabels } from "./reviewState";

function App() {
  const [diffView, setDiffView] = useState<DiffView>("diff");
  const [showSettings, setShowSettings] = useState(false);
  const [setupPage, setSetupPage] = useState<SetupPage>("downloads");
  const [setupBusy, setSetupBusy] = useState(false);
  const review = useReviewWorkspace(showSettings);
  const inspector = useInspectorResize();

  return (
    <div className="app-shell">
      <div className={`workspace ${review.showSetup || showSettings ? "setup-mode" : ""} ${inspector.resizing ? "is-resizing" : ""}`} style={inspector.style}>
        <ReviewSidebar
          showSetup={review.showSetup}
          setupPage={setupPage}
          showSettings={showSettings}
          textDirty={review.textDirty}
          status={review.status}
          hasDocument={review.document !== null}
          activeDecision={review.activeDecision}
          decisionLabel={`${review.activeDecisionOrigin}${decisionLabels[review.activeDecision]}`}
          busy={review.busy || setupBusy}
          projects={review.projects}
          projectId={review.projectId}
          selectedQueue={review.selectedQueue}
          totalItems={review.totalItems}
          cleanProgress={review.cleanProgress}
          ordinal={review.ordinal}
          pageInput={review.pageInput}
          statusRefresh={`${review.cleanProgress?.attempts ?? 0}:${review.cleanProgress?.manual_completed ?? 0}:${review.cleanProgress?.manual_llm_saved ?? 0}:${review.document?.document_review?.revision ?? 0}:${review.status}`}
          onShowSetupChange={(value) => { setShowSettings(false); review.setShowSetup(value); }}
          onSetupPageChange={(page) => { setShowSettings(false); setSetupPage(page); review.setShowSetup(true); }}
          onShowSettingsChange={(value) => { setShowSettings(value); if (value) review.setShowSetup(false); }}
          onProjectChange={review.changeProject}
          onPageInputChange={review.setPageInput}
          onCommitPage={review.commitPageInput}
          onNavigate={review.saveAndGo}
        />

        {showSettings ? <SettingsPage /> : <DocumentWorkspace
          showSetup={review.showSetup}
          setupPage={setupPage}
          projects={review.projects}
          document={review.document}
          draftBlocks={review.draftBlocks}
          editorText={review.editorText}
          llmCleaning={review.llmCleaning}
          llmResult={review.llmResult}
          onLlmClean={() => void review.cleanCurrentDocument()}
          textDirty={review.textDirty}
          busy={review.busy}
          activeBlockId={review.activeBlockId}
          onFinishSetup={review.finishSetup}
          onSetupPageChange={setSetupPage}
          onSetupBusyChange={setSetupBusy}
          onShowSetup={() => { setSetupPage("downloads"); review.setShowSetup(true); }}
          onActiveBlockChange={review.setActiveBlockId}
          onBlocksChange={review.updateDraftBlocks}
        />}

        {!review.showSetup && !showSettings && <div className="inspector-resizer" {...inspector.separatorProps} />}
        {!review.showSetup && !showSettings && <ReviewInspector
          document={review.document}
          editorText={review.editorText}
          diffView={diffView}
          onDiffViewChange={setDiffView}
        />}
      </div>
    </div>
  );
}

export default App;
