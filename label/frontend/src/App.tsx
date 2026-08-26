import DocumentWorkspace from "./components/DocumentWorkspace";
import ReviewInspector from "./components/ReviewInspector";
import ReviewSidebar from "./components/ReviewSidebar";
import { useReviewWorkspace } from "./hooks/useReviewWorkspace";
import { decisionLabels } from "./reviewState";

function App() {
  const review = useReviewWorkspace();

  return (
    <div className="app-shell">
      <div className={`workspace ${review.showSetup ? "setup-mode" : ""}`}>
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
          textDirty={review.textDirty}
          busy={review.busy}
          activeBlockId={review.activeBlockId}
          onFinishSetup={review.finishSetup}
          onShowSetup={() => review.setShowSetup(true)}
          onActiveBlockChange={review.setActiveBlockId}
          onBlocksChange={review.updateDraftBlocks}
        />

        {!review.showSetup && <ReviewInspector
          document={review.document}
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
