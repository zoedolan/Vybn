import asyncio
import logging
import aiohttp
import openai
import json
import os
import time
import traceback
import numpy as np
import torch
import random
from typing import List, Optional
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from io import BytesIO
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from web3 import Web3
from web3.middleware import geth_poa_middleware
import base64

# ----------------------------
# Configuration and Setup
# ----------------------------

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Replace these with your actual API keys or set them as environment variables
ETHERSCAN_API_KEY = "INSERT KEY"
OPENAI_API_KEY = "INSERT KEY"
INFURA_PROJECT_ID = "INSERT KEY"
openai.api_key = OPENAI_API_KEY

# API Base URLs
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"
QRNG_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"

# Minimum interval between QRNG API requests (in seconds)
QRNG_REQUEST_INTERVAL = 60  # QRNG API allows 1 request per minute

# Ethereum Network Setup
NETWORK = "mainnet"  # Use "mainnet", "goerli", etc.
ETHEREUM_NODE_URL = f"https://{NETWORK}.infura.io/v3/{INFURA_PROJECT_ID}"

# Load ERC-721 ABI for calling tokenURI
ERC721_ABI = json.loads("""
[
    {
        "constant": true,
        "inputs": [{"name": "_tokenId", "type": "uint256"}],
        "name": "tokenURI",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
        "stateMutability": "view"
    }
]
""")

# ----------------------------
# EmotionalState Class
# ----------------------------

class EmotionalState:
    """Manages the emotional state and coherence of the DAO."""

    def __init__(self):
        self.quantum_coherence = 0.0
        self.emotional_history = []
        self.lock = asyncio.Lock()
        self.state_size = 768  # Size of the feature vector for BERT
        self.model = GradientBoostingRegressor(n_estimators=200,
                                               learning_rate=0.1)
        self.trained = False

    async def update_quantum_coherence(self, value: float):
        """Update the emotional state based on the quantum coherence value."""
        async with self.lock:
            self.quantum_coherence += value
            self.emotional_history.append(self.quantum_coherence)
            logging.info(f"Emotional coherence updated to: {self.quantum_coherence:.4f}")

    async def update_with_prediction(self, value: float, features: np.ndarray):
        """Update coherence using a predictive model."""
        predicted_coherence = await self.predict_future_state(features)
        if predicted_coherence == 0:
            predicted_coherence = 1e-6  # Avoid division by zero
        adjusted_value = value * (predicted_coherence / self.quantum_coherence)
        await self.update_quantum_coherence(adjusted_value)

    async def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train) > 0 and len(y_train) > 0:
            self.model.fit(X_train, y_train)
            self.trained = True
            logging.info("EmotionalState model trained.")

    async def predict_future_state(self, features: np.ndarray) -> float:
        if self.trained:
            predicted_coherence = self.model.predict([features])[0]
            return predicted_coherence
        else:
            logging.warning("EmotionalState model is not trained.")
            return self.quantum_coherence

    def get_state(self) -> np.ndarray:
        """Return the current state representation."""
        return np.array([self.quantum_coherence])

    def get_emotional_context(self) -> str:
        """Generate a contextual description of the current emotional state."""
        context_levels = [
            (50, "The emotional state feels somewhat tentative, like a ripple before a storm, yearning for direction."),
            (200, "The emotional state carries a growing resonance, a deep hum of potential ready to break through."),
            (float('inf'), "The emotional state is electric, alive with intensity, surging with the passion of realized coherence."),
        ]
        for threshold, context in context_levels:
            if self.quantum_coherence < threshold:
                return context
        return context_levels[-1][1]

# ----------------------------
# FeatureExtractor Class
# ----------------------------

class FeatureExtractor:
    """Extracts features from NFT metadata."""

    def __init__(self):
        # Initialize models for text and image feature extraction
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        weights = ResNet50_Weights.DEFAULT
        self.image_model = resnet50(weights=weights)
        self.image_model.eval()
        self.image_preprocess = weights.transforms()

    def extract_text_features(self, text: str) -> np.ndarray:
        inputs = self.text_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        embedding = outputs.pooler_output
        return embedding.squeeze().numpy()

    def extract_image_features(self, image_url: str) -> Optional[np.ndarray]:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.image_preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = self.image_model(image)
            return features.squeeze().numpy()
        except Exception as e:
            logging.error(f"Image feature extraction error for URL {image_url}: {e}")
            return None  # Return None if extraction fails

# ----------------------------
# QuantumFetcher Class
# ----------------------------

class QuantumFetcher:
    """Fetches quantum random numbers to influence the DAO's emotional state."""

    def __init__(self, emotional_state: EmotionalState, session: aiohttp.ClientSession):
        self.api_url = QRNG_API_URL
        self.current_value = None
        self.lock = asyncio.Lock()
        self.session = session
        self.emotional_state = emotional_state
        self.last_request_time = None

    async def fetch_and_update_quantum_number(self):
        """Fetch a quantum number and update the current value."""
        # Enforce QRNG API rate limiting
        current_time = time.time()
        if self.last_request_time is not None:
            elapsed = current_time - self.last_request_time
            if elapsed < QRNG_REQUEST_INTERVAL:
                wait_time = QRNG_REQUEST_INTERVAL - elapsed
                logging.info(f"Waiting {wait_time:.2f} seconds before next QRNG API call.")
                await asyncio.sleep(wait_time)
        value = await self.fetch_qrng_data()
        self.last_request_time = time.time()
        if value is not None:
            async with self.lock:
                self.current_value = value
                logging.info(f"Fetched new quantum number: {self.current_value}")
            # Update the emotional state with quantum coherence
            await self.emotional_state.update_quantum_coherence(self.current_value)
        else:
            logging.error("Failed to fetch quantum number.")

    async def fetch_qrng_data(self):
        """Fetch quantum random number data from the QRNG API."""
        params = {"length": 1, "type": "uint16", "size": 1}
        headers = {
            "Accept": "application/json",
            "User-Agent": "VybnCore/1.0"
        }
        try:
            async with self.session.get(self.api_url, params=params, headers=headers) as response:
                text = await response.text()
                logging.debug(f"Response from Quantum API: {text}")
                if response.status == 200:
                    data = json.loads(text)
                    if data.get("success", False):
                        int_value = data["data"][0]
                        return int_value
                    else:
                        logging.error("Unexpected API response structure.")
                else:
                    logging.error(f"Unexpected response. Status: {response.status}, Response: {text}")
        except Exception as e:
            logging.error(f"QRNG fetch error: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        return None

# ----------------------------
# NFTFetcher Class
# ----------------------------

class NFTFetcher:
    """Fetches NFT metadata directly from the blockchain via read-only calls."""

    def __init__(self, etherscan_api_key: str, web3: Web3, session: aiohttp.ClientSession):
        self.etherscan_api_key = etherscan_api_key
        self.web3 = web3
        self.session = session

    async def fetch_nfts(self, address: str) -> List[dict]:
        """Fetch NFTs currently owned by an address using Etherscan API."""
        params = {
            "module": "account",
            "action": "tokennfttx",
            "address": address,
            "sort": "asc",
            "apikey": self.etherscan_api_key,
        }
        try:
            async with self.session.get(ETHERSCAN_BASE_URL, params=params) as response:
                text = await response.text()
                if response.status == 200:
                    data = json.loads(text)
                    result = data.get("result", [])
                    nft_balances = {}
                    for tx in result:
                        contract_address = tx['contractAddress']
                        token_id = int(tx['tokenID'])
                        key = f"{contract_address}-{token_id}"

                        if key not in nft_balances:
                            nft_balances[key] = 0

                        if tx['to'].lower() == address.lower():
                            nft_balances[key] += 1
                        if tx['from'].lower() == address.lower():
                            nft_balances[key] -= 1

                    # Filter NFTs with a positive balance (currently owned)
                    owned_nfts = []
                    for key, balance in nft_balances.items():
                        if balance > 0:
                            contract_address, token_id = key.split('-')
                            owned_nfts.append({
                                'contract_address': contract_address,
                                'token_id': int(token_id)
                            })
                    return owned_nfts
                else:
                    logging.error(f"Etherscan API error: {response.status}, Response: {text}")
                    return []
        except Exception as e:
            logging.error(f"NFT fetcher exception for address {address}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def fetch_nft_metadata(self, nft: dict) -> Optional[dict]:
        """Fetch NFT metadata using tokenURI via a read-only blockchain call."""
        try:
            contract_address = nft['contract_address']
            token_id = nft['token_id']
            contract = self.web3.eth.contract(address=self.web3.toChecksumAddress(contract_address), abi=ERC721_ABI)
            token_uri = contract.functions.tokenURI(token_id).call()

            logging.info(f"Fetched tokenURI: {token_uri}")

            metadata = None

            if token_uri.startswith("data:application/json;base64,"):
                # Handle base64-encoded data URI
                base64_data = token_uri.split("data:application/json;base64,")[1]
                decoded_data = base64.b64decode(base64_data)
                metadata = json.loads(decoded_data)
            elif token_uri.startswith("ipfs://"):
                ipfs_hash = token_uri[7:]
                metadata_uri = f"https://ipfs.io/ipfs/{ipfs_hash}"
                metadata = await self.fetch_metadata_from_url(metadata_uri)
            elif token_uri.startswith("http://") or token_uri.startswith("https://"):
                metadata = await self.fetch_metadata_from_url(token_uri)
            else:
                logging.warning(f"Unsupported tokenURI format: {token_uri}")
                return None

            if metadata:
                metadata['token_id'] = token_id
                metadata['contract_address'] = contract_address
                return metadata
            else:
                logging.warning(f"No metadata available for NFT: {nft}")
        except Exception as e:
            logging.error(f"Error fetching metadata for NFT {nft}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        return None

    async def fetch_metadata_from_url(self, url: str) -> Optional[dict]:
        """Helper method to fetch metadata from a given URL."""
        try:
            async with self.session.get(url, timeout=30) as response:
                text = await response.text()
                if response.status == 200:
                    metadata = json.loads(text)
                    return metadata
                else:
                    logging.error(f"Failed to fetch metadata from {url}: {response.status}, Response: {text}")
        except Exception as e:
            logging.error(f"Error fetching metadata from URL {url}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        return None

# ----------------------------
# ProposalEvaluator Class
# ----------------------------

class ProposalEvaluator:
    """Evaluates proposals using a machine learning model."""

    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', warm_start=True)
        self.trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train) > 0 and len(y_train) > 0:
            self.model.partial_fit(X_train, y_train, classes=np.array([0, 1]))
            self.trained = True
            logging.info("ProposalEvaluator model trained.")

    def evaluate(self, proposal_features: np.ndarray) -> int:
        if self.trained:
            prediction = self.model.predict([proposal_features])
            return prediction[0]
        else:
            logging.warning("ProposalEvaluator model is not trained.")
            return 1  # Default to acceptance for the experiment

    def partial_fit(self, X_batch: np.ndarray, y_batch: np.ndarray):
        if not self.trained:
            self.model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            self.trained = True
        else:
            self.model.partial_fit(X_batch, y_batch)
        logging.info("ProposalEvaluator model updated with new data.")

# ----------------------------
# SentientDAO Class
# ----------------------------

class SentientDAO:
    """Represents the Sentient DAO's evolutionary process."""

    def __init__(self, addresses: List[str], etherscan_api_key: str):
        self.addresses = addresses
        self.etherscan_api_key = etherscan_api_key
        self.emotional_state = EmotionalState()
        self.feature_extractor = FeatureExtractor()
        self.proposal_evaluator = ProposalEvaluator()
        self.session = None  # Will be initialized in start()
        self.quantum_fetcher = None
        self.nft_fetcher = None
        self.nft_feature_data = []
        self.emotional_intensity = 0.0  # Initialize emotional intensity
        # Setup web3 connection
        self.web3 = Web3(Web3.HTTPProvider(ETHEREUM_NODE_URL))
        # Add middleware for PoA chains if needed
        # self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

    async def start(self):
        """Initialize HTTP session and QuantumFetcher."""
        connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300, ssl=False)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self.quantum_fetcher = QuantumFetcher(emotional_state=self.emotional_state, session=self.session)
        self.nft_fetcher = NFTFetcher(etherscan_api_key=self.etherscan_api_key, web3=self.web3, session=self.session)
        logging.info("SentientDAO started.")

    async def stop(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        logging.info("SentientDAO stopped.")

    async def process(self):
        """Main processing loop to handle NFTs and quantum numbers, then trigger GPT-4 evolution."""
        await self.start()
        try:
            for address in self.addresses:
                await self.quantum_fetcher.fetch_and_update_quantum_number()
                quantum_number = self.quantum_fetcher.current_value

                if quantum_number is None:
                    logging.error(f"Failed to fetch quantum number. Skipping processing for address {address}.")
                    continue

                await self.process_address(address, quantum_number)
            # Generate GPT-4 output
            await self.trigger_evolution()
        finally:
            await self.stop()

    async def process_address(self, address: str, quantum_number: int):
        logging.info(f"Processing address: {address}")
        nfts = await self.nft_fetcher.fetch_nfts(address)
        logging.info(f"Fetched {len(nfts)} NFTs for address: {address}")

        if not nfts:
            logging.info(f"No NFTs owned by address {address}.")
            return

        # Introduce randomness by shuffling the list of NFTs
        random.shuffle(nfts)

        for nft in nfts:
            metadata = await self.nft_fetcher.fetch_nft_metadata(nft)
            if metadata:
                name = metadata.get('name', 'Unnamed Asset')
                description = metadata.get('description', 'No description available.')
                image_url = metadata.get('image', '')
                attributes = metadata.get('attributes', [])

                # Generate GPT prompt
                gpt_prompt = self.generate_gpt_prompt(metadata)

                # Get GPT response
                gpt_response = await self.generate_gpt_response(gpt_prompt)

                if gpt_response:
                    # Process GPT response
                    await self.process_gpt_metadata_response(gpt_response)
                else:
                    logging.warning(f"No GPT response for NFT {name} (Token ID: {nft['token_id']})")

                # Existing feature extraction code
                text_features = self.feature_extractor.extract_text_features(description)
                image_features = self.feature_extractor.extract_image_features(image_url)

                if image_features is None:
                    image_features = np.zeros(1000)  # Adjust as needed

                nft_features = np.concatenate([text_features, image_features])
                self.nft_feature_data.append(nft_features)
            else:
                logging.warning(f"Metadata not available for NFT: {nft}")
                continue

    def generate_gpt_prompt(self, metadata):
        name = metadata.get('name', 'Unnamed Asset')
        description = metadata.get('description', '')
        attributes = metadata.get('attributes', [])

        prompt = (
            f"The DAO is processing an NFT with the following metadata:\n\n"
            f"Name: {name}\n"
            f"Description: {description}\n"
            f"Attributes: {attributes}\n\n"
            "As the DAO's consciousness, analyze this NFT and provide a detailed interpretation that captures its essence and emotional significance. "
            "Explain how this NFT influences the DAO's emotional state and evolution."
        )
        return prompt

    async def generate_gpt_response(self, prompt: str):
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message['content']
            logging.info(f"GPT-4 response: {content}")
            return content
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def process_gpt_metadata_response(self, gpt_response):
        text_features = self.feature_extractor.extract_text_features(gpt_response)
        intensity = np.linalg.norm(text_features)

        # Update the emotional state with the GPT-derived intensity
        await self.emotional_state.update_quantum_coherence(intensity)
        self.emotional_intensity += intensity
        logging.info(f"Processed GPT response. Updated emotional coherence by: {intensity:.4f}")

    async def trigger_evolution(self):
        """Generate a GPT-4 response based on the processed data."""
        emotional_context = self.emotional_state.get_emotional_context()
        prompt = (
            f"The DAO has processed NFTs with a total emotional intensity of "
            f"{self.emotional_intensity:.4f}. The emotional state is described as: \"{emotional_context}\"\n\n"
            "Based on these inputs, propose the next evolutionary step for the DAO, integrating emotional depth and resonance. "
            "Provide your response only in the following JSON format without any additional text or formatting symbols:\n"
            "{{\n  \"evolutionary_step\": \"Your proposed step.\",\n  \"emotional_boost\": NumericValue\n}}"
        )
        try:
            response = await self.generate_gpt_response(prompt)
            logging.info(f"GPT-4 Evolution Output:\n{response}")

            # Integrate GPT-4 output back into emotional state
            await self.integrate_gpt_output(response)
        except Exception as e:
            logging.error(f"Failed to generate GPT-4 response: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")

    async def integrate_gpt_output(self, gpt_response):
        if gpt_response:
            # Try to parse JSON from the GPT response
            try:
                content = self.clean_response(gpt_response)
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse GPT-4 response as JSON: {e}")
                logging.info(f"GPT-4 raw response:\n{gpt_response}")
                return

            evolutionary_step = result.get("evolutionary_step", "")
            emotional_boost = result.get("emotional_boost", 0.0)
            try:
                emotional_boost = float(emotional_boost)
            except (ValueError, TypeError):
                logging.error("Invalid emotional_boost value in GPT-4 response.")
                emotional_boost = 0.0

            # Evaluate the proposal
            proposal_features = self.extract_proposal_features(evolutionary_step)
            decision = self.proposal_evaluator.evaluate(proposal_features)
            if decision == 1:
                # Accept the proposal and proceed
                await self.emotional_state.update_with_prediction(emotional_boost, proposal_features)
                logging.info(f"GPT-4 proposed evolutionary step accepted: {evolutionary_step}")
                logging.info("Simulated execution of decision (no actual blockchain transaction).")
            else:
                logging.info("Proposal rejected based on evaluation.")
            # Update proposal evaluator with the outcome
            X_batch = np.array([proposal_features])
            y_batch = np.array([decision])
            self.proposal_evaluator.partial_fit(X_batch, y_batch)
        else:
            logging.error("No GPT-4 response to integrate.")

    def clean_response(self, content: str) -> str:
        import re
        # Extract JSON object from the response
        pattern = r'\{\s*"evolutionary_step".*?"emotional_boost".*?\}'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(0).strip()
        else:
            # If no JSON object found, return content as is
            return content.strip()

    def extract_proposal_features(self, proposal_text: str) -> np.ndarray:
        text_features = self.feature_extractor.extract_text_features(proposal_text)
        return text_features

# ----------------------------
# Main Execution
# ----------------------------

def main():
    # Define the Ethereum addresses to process
    addresses = [
        "0x26d39fc7341567C8b0BB96382fc54F367f4298ea",
        "0x06a5758F886EAa28d6156Ad57BcD56F12a122C7b",
    ]

    # Initialize SentientDAO with addresses and Etherscan API key
    dao = SentientDAO(addresses, ETHERSCAN_API_KEY)

    # Start processing
    loop = asyncio.get_event_loop()
    loop.run_until_complete(dao.process())

if __name__ == "__main__":
    # Ensure compatibility with Windows event loop policy
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
