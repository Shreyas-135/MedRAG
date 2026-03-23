// deploy.js – deploys FederatedAggregator to the running Hardhat node
// Usage: npx hardhat run scripts/deploy.js --network localhost

const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const balance = await deployer.provider.getBalance(deployer.address);

  console.log("Deploying FederatedAggregator with account:", deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH");

  const FederatedAggregator = await hre.ethers.getContractFactory("FederatedAggregator");
  const aggregator = await FederatedAggregator.deploy();
  await aggregator.waitForDeployment();

  const address = await aggregator.getAddress();
  console.log("\nFederatedAggregator deployed to:", address);
  console.log("\nTo use with Python, set the environment variable:");
  console.log("  Windows PowerShell:");
  console.log("    $env:AGGREGATOR_CONTRACT_ADDRESS=\"" + address + "\"");
  console.log("  Linux/macOS:");
  console.log("    export AGGREGATOR_CONTRACT_ADDRESS=" + address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
