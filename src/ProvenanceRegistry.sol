// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ProvenanceRegistry
 * @notice Records cryptographic provenance anchors for RAG inference citations.
 *
 * Each anchor captures hashes only (no raw text) for a single inference:
 *   - bundleHash   : sha256 of the canonical provenance bundle
 *   - modelHash    : hash identifying the model version
 *   - kbHash       : hash of the knowledge base at inference time
 *   - explanationHash : hash of the generated explanation
 *   - signer       : Ethereum address that signed the bundle (MetaMask)
 *   - timestamp    : block.timestamp at anchoring
 *
 * Compatible with existing Aggregator.sol deployment; deploy this contract
 * separately and reference it from ProvenanceIntegrator.
 */
contract ProvenanceRegistry {

    event ProvenanceAnchored(
        bytes32 indexed bundleHash,
        bytes32 modelHash,
        bytes32 kbHash,
        bytes32 explanationHash,
        address indexed signer,
        uint256 timestamp
    );

    struct Anchor {
        bytes32 modelHash;
        bytes32 kbHash;
        bytes32 explanationHash;
        address signer;
        uint256 timestamp;
        bool exists;
    }

    mapping(bytes32 => Anchor) public anchors;

    /**
     * @notice Anchor a provenance bundle on-chain.
     * @param bundleHash     SHA-256 of the canonical provenance bundle (bytes32).
     * @param modelHash      Hash of model version.
     * @param kbHash         Hash of knowledge base.
     * @param explanationHash Hash of the explanation text.
     * @param signer         Address of the MetaMask signer.
     */
    function anchorProvenance(
        bytes32 bundleHash,
        bytes32 modelHash,
        bytes32 kbHash,
        bytes32 explanationHash,
        address signer
    ) external {
        anchors[bundleHash] = Anchor({
            modelHash: modelHash,
            kbHash: kbHash,
            explanationHash: explanationHash,
            signer: signer,
            timestamp: block.timestamp,
            exists: true
        });

        emit ProvenanceAnchored(
            bundleHash,
            modelHash,
            kbHash,
            explanationHash,
            signer,
            block.timestamp
        );
    }

    /**
     * @notice Check whether a bundle hash has been anchored.
     */
    function isAnchored(bytes32 bundleHash) external view returns (bool) {
        return anchors[bundleHash].exists;
    }

    /**
     * @notice Retrieve anchor details for a given bundle hash.
     */
    function getAnchor(bytes32 bundleHash) external view returns (
        bytes32 modelHash,
        bytes32 kbHash,
        bytes32 explanationHash,
        address signer,
        uint256 timestamp
    ) {
        Anchor storage a = anchors[bundleHash];
        require(a.exists, "ProvenanceRegistry: bundle not anchored");
        return (a.modelHash, a.kbHash, a.explanationHash, a.signer, a.timestamp);
    }
}
