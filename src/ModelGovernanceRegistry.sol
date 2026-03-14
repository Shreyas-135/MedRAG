// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ModelGovernanceRegistry
 * @notice Multi-signature governance contract for promoting trained model
 *         versions to APPROVED status. Requires 3-of-4 hospital approvals
 *         (threshold configurable at deployment).
 *
 * Only hashes are stored on-chain; no raw weights or medical data.
 *
 * Workflow:
 *   1. Admin (deployer) calls registerModel(modelHash, metadataHash).
 *   2. Each hospital calls approveModel(modelHash) once.
 *   3. When approvalCount >= requiredApprovals the model is marked APPROVED
 *      and a ModelApproved event is emitted.
 */
contract ModelGovernanceRegistry {

    // ------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------

    event ModelRegistered(
        bytes32 indexed modelHash,
        bytes32 metadataHash,
        address indexed registeredBy,
        uint256 timestamp
    );

    event ModelApprovalRecorded(
        bytes32 indexed modelHash,
        address indexed hospital,
        uint256 approvalCount,
        uint256 timestamp
    );

    event ModelApproved(
        bytes32 indexed modelHash,
        uint256 timestamp
    );

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------

    enum Status { UNKNOWN, PENDING, APPROVED }

    struct ModelRecord {
        bytes32 metadataHash;
        address registeredBy;
        uint256 registeredAt;
        uint256 approvalCount;
        Status  status;
        bool    exists;
    }

    /// @notice Minimum approvals required to mark a model APPROVED.
    uint256 public immutable requiredApprovals;

    /// @notice Address that deployed the contract (can register models).
    address public immutable admin;

    /// @notice Model records keyed by model hash.
    mapping(bytes32 => ModelRecord) public models;

    /// @notice Per-model per-hospital approval flag (prevents double-approval).
    mapping(bytes32 => mapping(address => bool)) public hasApproved;

    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------

    /**
     * @param _requiredApprovals Number of distinct hospital approvals needed
     *        (e.g. 3 for a 3-of-4 scheme). Defaults to 3 if 0 is passed.
     */
    constructor(uint256 _requiredApprovals) {
        admin = msg.sender;
        requiredApprovals = _requiredApprovals == 0 ? 3 : _requiredApprovals;
    }

    // ------------------------------------------------------------------
    // Public functions
    // ------------------------------------------------------------------

    /**
     * @notice Register a new model version as PENDING.
     * @param modelHash     SHA-256 of the model weights/checkpoint (bytes32).
     * @param metadataHash  Optional SHA-256 of model metadata JSON (bytes32).
     *                      Pass bytes32(0) if not used.
     */
    function registerModel(bytes32 modelHash, bytes32 metadataHash) external {
        require(msg.sender == admin, "ModelGovernanceRegistry: only admin");
        require(!models[modelHash].exists, "ModelGovernanceRegistry: already registered");

        models[modelHash] = ModelRecord({
            metadataHash:  metadataHash,
            registeredBy:  msg.sender,
            registeredAt:  block.timestamp,
            approvalCount: 0,
            status:        Status.PENDING,
            exists:        true
        });

        emit ModelRegistered(modelHash, metadataHash, msg.sender, block.timestamp);
    }

    /**
     * @notice Submit an approval for a PENDING model.
     * @param modelHash SHA-256 of the model version to approve.
     */
    function approveModel(bytes32 modelHash) external {
        ModelRecord storage rec = models[modelHash];
        require(rec.exists, "ModelGovernanceRegistry: model not registered");
        require(rec.status == Status.PENDING, "ModelGovernanceRegistry: not pending");
        require(!hasApproved[modelHash][msg.sender], "ModelGovernanceRegistry: already approved");

        hasApproved[modelHash][msg.sender] = true;
        rec.approvalCount += 1;

        emit ModelApprovalRecorded(modelHash, msg.sender, rec.approvalCount, block.timestamp);

        if (rec.approvalCount >= requiredApprovals) {
            rec.status = Status.APPROVED;
            emit ModelApproved(modelHash, block.timestamp);
        }
    }

    // ------------------------------------------------------------------
    // View functions
    // ------------------------------------------------------------------

    /**
     * @notice Return the approval status of a model.
     * @return 0 = UNKNOWN, 1 = PENDING, 2 = APPROVED
     */
    function getStatus(bytes32 modelHash) external view returns (uint8) {
        return uint8(models[modelHash].status);
    }

    /**
     * @notice Return true if the model is APPROVED.
     */
    function isApproved(bytes32 modelHash) external view returns (bool) {
        return models[modelHash].status == Status.APPROVED;
    }

    /**
     * @notice Return the current approval count for a model.
     */
    function getApprovalCount(bytes32 modelHash) external view returns (uint256) {
        return models[modelHash].approvalCount;
    }
}
