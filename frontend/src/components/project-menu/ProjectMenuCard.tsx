import React from "react";
import { useDispatch } from "react-redux";
import toast from "react-hot-toast";
import posthog from "posthog-js";
import EllipsisH from "#/icons/ellipsis-h.svg?react";
import { ModalBackdrop } from "../modals/modal-backdrop";
import { ConnectToGitHubModal } from "../modals/connect-to-github-modal";
import { addUserMessage } from "#/state/chatSlice";
import { createChatMessage } from "#/services/chatService";
import { ProjectMenuCardContextMenu } from "./project.menu-card-context-menu";
import { ProjectMenuDetailsPlaceholder } from "./project-menu-details-placeholder";
import { ProjectMenuDetails } from "./project-menu-details";
import { downloadWorkspace } from "#/utils/download-workspace";
import { LoadingSpinner } from "../modals/LoadingProject";
import { useWsClient } from "#/context/ws-client-provider";

interface ProjectMenuCardProps {
  isConnectedToGitHub: boolean;
  githubData: {
    avatar: string;
    repoName: string;
    lastCommit: GitHubCommit;
  } | null;
}

export function ProjectMenuCard({
  isConnectedToGitHub,
  githubData,
}: ProjectMenuCardProps) {
  const { send } = useWsClient();
  const dispatch = useDispatch();

  const [contextMenuIsOpen, setContextMenuIsOpen] = React.useState(false);
  const [connectToGitHubModalOpen, setConnectToGitHubModalOpen] =
    React.useState(false);
  const [working, setWorking] = React.useState(false);

  const toggleMenuVisibility = () => {
    setContextMenuIsOpen((prev) => !prev);
  };

  const handlePushToGitHub = () => {
    posthog.capture("push_to_github_button_clicked");
    const rawEvent = {
      content: `
Please push the changes to GitHub and open a pull request.
`,
      imageUrls: [],
      timestamp: new Date().toISOString(),
    };
    const event = createChatMessage(
      rawEvent.content,
      rawEvent.imageUrls,
      rawEvent.timestamp,
    );

    send(event); // send to socket
    dispatch(addUserMessage(rawEvent)); // display in chat interface
    setContextMenuIsOpen(false);
  };

  const handleDownloadWorkspace = () => {
    posthog.capture("download_workspace_button_clicked");
    try {
      setWorking(true);
      downloadWorkspace().then(
        () => setWorking(false),
        () => setWorking(false),
      );
    } catch (error) {
      toast.error("Failed to download workspace");
    }
  };

  return (
    <div className="px-4 py-[10px] w-[337px] rounded-xl border border-[#525252] flex justify-between items-center relative">
      {!working && contextMenuIsOpen && (
        <ProjectMenuCardContextMenu
          isConnectedToGitHub={isConnectedToGitHub}
          onConnectToGitHub={() => setConnectToGitHubModalOpen(true)}
          onPushToGitHub={handlePushToGitHub}
          onDownloadWorkspace={handleDownloadWorkspace}
          onClose={() => setContextMenuIsOpen(false)}
        />
      )}
      {githubData && (
        <ProjectMenuDetails
          repoName={githubData.repoName}
          avatar={githubData.avatar}
          lastCommit={githubData.lastCommit}
        />
      )}
      {!githubData && (
        <ProjectMenuDetailsPlaceholder
          isConnectedToGitHub={isConnectedToGitHub}
          onConnectToGitHub={() => setConnectToGitHubModalOpen(true)}
        />
      )}
      <button
        type="button"
        onClick={toggleMenuVisibility}
        aria-label="Open project menu"
      >
        {working ? (
          <LoadingSpinner size="small" />
        ) : (
          <EllipsisH width={36} height={36} />
        )}
      </button>
      {connectToGitHubModalOpen && (
        <ModalBackdrop onClose={() => setConnectToGitHubModalOpen(false)}>
          <ConnectToGitHubModal
            onClose={() => setConnectToGitHubModalOpen(false)}
          />
        </ModalBackdrop>
      )}
    </div>
  );
}
