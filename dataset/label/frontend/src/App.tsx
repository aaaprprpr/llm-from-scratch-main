import DocumentWorkspace from "./components/DocumentWorkspace";
import ReviewInspector from "./components/ReviewInspector";
import ReviewSidebar from "./components/ReviewSidebar";
import { useReviewWorkspace } from "./hooks/useReviewWorkspace";
import { useInspectorResize } from "./hooks/useInspectorResize";
import { decisionLabels } from "./reviewState";

function App() {
  const review = useReviewWorkspace();
  const inspector = useInspectorResize();

  return (
    <div className="app-shell">
      <div className={`workspace ${review.showSetup ? "setup-mode" : ""} ${inspector.resizing ? "is-resizing" : ""}`} style={inspector.style}>
        <ReviewSidebar
          showSetup={review.showSetup}
          textDirty={review.textDirty}
          status={review.status}
          hasDocument={review.document !== null}
          activeDecision={review.activeDecision}
          decisionLabel={decisionLabels[review.activeDecision]}
          busy={review.busy}
          projects={review.projects}
          projectId={review.projectId}
          selectedQueue={review.selectedQueue}
          totalItems={review.totalItems}
          ordinal={review.ordinal}
          pageInput={review.pageInput}
          onShowSetupChange={review.setShowSetup}
          onSimplify={() => void review.simplifyCurrentDocument()}
          onProjectChange={review.changeProject}
          onPageInputChange={review.setPageInput}
          onCommitPage={review.commitPageInput}
          onNavigate={review.saveAndGo}
        />

        <DocumentWorkspace
          showSetup={review.showSetup}
          projects={review.projects}
          projectId={review.projectId}
          document={review.document}
          draftBlocks={review.draftBlocks}
          editorText={review.editorText}
          llmCleaning={review.llmCleaning}
          llmResult={review.llmResult}
          onLlmClean={() => void review.cleanCurrentDocument()}
          onUndoLlmClean={review.undoDraft}
          textDirty={review.textDirty}
          busy={review.busy}
          activeBlockId={review.activeBlockId}
          onFinishSetup={review.finishSetup}
          onShowSetup={() => review.setShowSetup(true)}
          onActiveBlockChange={review.setActiveBlockId}
          onBlocksChange={review.updateDraftBlocks}
        />

        {!review.showSetup && <div className="inspector-resizer" {...inspector.separatorProps} />}
        {!review.showSetup && <ReviewInspector
          document={review.document}
          editorText={review.editorText}
          quality={review.quality}
          category={review.category}
          notes={review.notes}
          onQualityChange={review.setQuality}
          onCategoryChange={review.setCategory}
          onNotesChange={review.setNotes}
        />}
      </div>
    </div>
  );
}

export default App;
