// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FederatedKBRegistry
 * @notice On-chain registry for federated knowledge base contribution commitments.
 *
 * @dev Each Flower federated learning round produces a SHA-256 commitment
 *      hash over all KB entries added during that round
 *      (computed by FedKBManager.get_round_commitment() in Python).
 *      Anchoring this hash on-chain creates an immutable, tamper-evident
 *      audit trail that any hospital or regulator can independently verify.
 *
 *      Only hashes are stored — no embeddings, no patient data, no text.
 *
 * Roles
 * -----
 * - owner   : Deploying address; can register new hospitals and set the
 *             minimum required contributors.
 * - hospital : Registered participant; may submit round commitments.
 *
 * Workflow
 * --------
 * 1. Owner deploys and registers hospital addresses via registerHospital().
 * 2. After each FL round the server computes a commitment hash in Python and
 *    calls submitRoundCommitment() from one of the registered hospital wallets.
 * 3. Anyone can call getCommitment() to retrieve the stored hash and verify
 *    it independently by re-running FedKBManager.get_round_commitment().
 * 4. isVerified() checks whether a given hash matches what is on-chain.
 */
contract FederatedKBRegistry {

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    struct RoundCommitment {
        bytes32  commitmentHash;   // SHA-256 of all KB entries added this round
        address  submittedBy;      // Hospital address that anchored the commitment
        uint256  timestamp;        // Block timestamp of submission
        uint256  numContributors;  // Number of hospitals that contributed this round
        bool     exists;
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    address public owner;

    /// @notice Registered hospital addresses eligible to submit commitments
    mapping(address => bool) public registeredHospitals;
    address[] public hospitalList;

    /// @notice round_id => commitment record
    mapping(uint256 => RoundCommitment) public commitments;
    uint256[] public roundIds;

    uint256 public minContributors;  // Minimum hospitals needed per round

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    event HospitalRegistered(address indexed hospital, uint256 timestamp);
    event RoundCommitmentAnchored(
        uint256 indexed roundId,
        bytes32  commitmentHash,
        address  indexed submittedBy,
        uint256  numContributors,
        uint256  timestamp
    );

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "FederatedKBRegistry: caller is not owner");
        _;
    }

    modifier onlyHospital() {
        require(
            registeredHospitals[msg.sender],
            "FederatedKBRegistry: caller is not a registered hospital"
        );
        _;
    }

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * @param _minContributors Minimum number of hospital contributors required
     *        before a round commitment may be submitted.  Prevents a single
     *        hospital from unilaterally anchoring a commitment.
     */
    constructor(uint256 _minContributors) {
        owner = msg.sender;
        minContributors = _minContributors;
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    /**
     * @notice Register a hospital address as an eligible contributor.
     * @param hospital Ethereum address of the hospital node.
     */
    function registerHospital(address hospital) external onlyOwner {
        require(hospital != address(0), "FederatedKBRegistry: zero address");
        require(!registeredHospitals[hospital], "FederatedKBRegistry: already registered");
        registeredHospitals[hospital] = true;
        hospitalList.push(hospital);
        emit HospitalRegistered(hospital, block.timestamp);
    }

    /**
     * @notice Update the minimum contributor threshold.
     * @param _minContributors New minimum value (must be >= 1).
     */
    function setMinContributors(uint256 _minContributors) external onlyOwner {
        require(_minContributors >= 1, "FederatedKBRegistry: must be >= 1");
        minContributors = _minContributors;
    }

    // -----------------------------------------------------------------------
    // Commitment submission
    // -----------------------------------------------------------------------

    /**
     * @notice Anchor the federated KB commitment hash for a completed round.
     *
     * @dev Called by the server (or one of the hospital nodes) after
     *      FedKBManager.aggregate_round() finishes in Python.
     *
     * @param roundId          The Flower federated learning round number.
     * @param commitmentHash   SHA-256 hash returned by
     *                         FedKBManager.get_round_commitment(roundId).
     * @param numContributors  Number of hospitals that contributed this round
     *                         (must be >= minContributors).
     */
    function submitRoundCommitment(
        uint256 roundId,
        bytes32 commitmentHash,
        uint256 numContributors
    ) external onlyHospital {
        require(!commitments[roundId].exists, "FederatedKBRegistry: round already anchored");
        require(
            numContributors >= minContributors,
            "FederatedKBRegistry: not enough contributors"
        );
        // Reject the zero hash (bytes32(0)) to prevent accidental empty submissions.
        // This is coordinated with the Python-side FedKBManager.get_round_commitment(),
        // which returns None (not a hash) when a round has no contributions — callers
        // must check for None before calling this function.
        require(commitmentHash != bytes32(0), "FederatedKBRegistry: empty hash");

        commitments[roundId] = RoundCommitment({
            commitmentHash:  commitmentHash,
            submittedBy:     msg.sender,
            timestamp:       block.timestamp,
            numContributors: numContributors,
            exists:          true
        });
        roundIds.push(roundId);

        emit RoundCommitmentAnchored(
            roundId,
            commitmentHash,
            msg.sender,
            numContributors,
            block.timestamp
        );
    }

    // -----------------------------------------------------------------------
    // Read helpers
    // -----------------------------------------------------------------------

    /**
     * @notice Retrieve the anchored commitment for a round.
     * @param roundId FL round number.
     * @return commitmentHash  The stored SHA-256 commitment hash.
     * @return submittedBy     Address that anchored it.
     * @return timestamp       Block timestamp of anchoring.
     * @return numContributors Number of contributing hospitals.
     */
    function getCommitment(uint256 roundId)
        external
        view
        returns (
            bytes32 commitmentHash,
            address submittedBy,
            uint256 timestamp,
            uint256 numContributors
        )
    {
        require(commitments[roundId].exists, "FederatedKBRegistry: round not found");
        RoundCommitment storage c = commitments[roundId];
        return (c.commitmentHash, c.submittedBy, c.timestamp, c.numContributors);
    }

    /**
     * @notice Check whether a given hash matches the on-chain commitment for
     *         a round.  Allows independent verification without off-chain
     *         storage.
     * @param roundId FL round number.
     * @param hash    SHA-256 hash to verify.
     * @return True if the hash matches the stored commitment.
     */
    function isVerified(uint256 roundId, bytes32 hash) external view returns (bool) {
        if (!commitments[roundId].exists) return false;
        return commitments[roundId].commitmentHash == hash;
    }

    /**
     * @notice Return the total number of anchored rounds.
     */
    function getRoundCount() external view returns (uint256) {
        return roundIds.length;
    }

    /**
     * @notice Return the list of all registered hospital addresses.
     */
    function getHospitals() external view returns (address[] memory) {
        return hospitalList;
    }
}
