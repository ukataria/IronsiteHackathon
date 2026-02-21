import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// Application state
const state = {
    timeline: null,
    currentSegment: 0,
    pointClouds: [],
    changeCloudsPairs: [],
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    currentMesh: null,
    referenceMesh: null,
    addedMesh: null,
    removedMesh: null,
};

// Initialize Three.js scene
function initScene() {
    const canvas = document.getElementById('canvas');
    const container = document.getElementById('canvas-container');

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x1a1a1a);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    state.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    state.camera.position.set(5, 5, 5);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    state.renderer.setSize(container.clientWidth, container.clientHeight);
    state.renderer.setPixelRatio(window.devicePixelRatio);

    // Controls
    state.controls = new OrbitControls(state.camera, canvas);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    state.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    state.scene.add(directionalLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    state.scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    state.scene.add(axesHelper);

    // Handle window resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        state.camera.aspect = width / height;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(width, height);
    });
}

// Load timeline JSON
async function loadTimeline() {
    try {
        const response = await fetch('timeline.json');
        if (!response.ok) {
            throw new Error('Timeline not found');
        }
        state.timeline = await response.json();
        console.log('Timeline loaded:', state.timeline);
        return state.timeline;
    } catch (error) {
        console.error('Failed to load timeline:', error);
        alert('Error loading timeline.json. Make sure viewer assets are properly exported.');
        return null;
    }
}

// Load a single point cloud
function loadPointCloud(path) {
    return new Promise((resolve, reject) => {
        const loader = new PLYLoader();
        loader.load(
            `assets/${path}`,
            (geometry) => {
                console.log(`Loaded: ${path} (${geometry.attributes.position.count} points)`);
                resolve(geometry);
            },
            (progress) => {
                // Progress callback
            },
            (error) => {
                console.error(`Failed to load ${path}:`, error);
                reject(error);
            }
        );
    });
}

// Load all point clouds
async function loadAllPointClouds() {
    const loadingEl = document.getElementById('loading');
    loadingEl.classList.remove('hidden');

    try {
        // Load segment point clouds
        for (const segment of state.timeline.segments) {
            const geometry = await loadPointCloud(segment.pointcloud);
            state.pointClouds.push(geometry);

            // Load change clouds if available
            const changes = segment.changes;
            const changePair = { added: null, removed: null };

            if (changes && changes.added_cloud) {
                try {
                    changePair.added = await loadPointCloud(changes.added_cloud);
                } catch (e) {
                    console.warn(`Could not load added cloud for segment ${segment.id}`);
                }
            }

            if (changes && changes.removed_cloud) {
                try {
                    changePair.removed = await loadPointCloud(changes.removed_cloud);
                } catch (e) {
                    console.warn(`Could not load removed cloud for segment ${segment.id}`);
                }
            }

            state.changeCloudsPairs.push(changePair);
        }

        // Load reference cloud
        try {
            const refGeometry = await loadPointCloud('reference.ply');
            const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, opacity: 0.3, transparent: true });
            state.referenceMesh = new THREE.Points(refGeometry, material);
            state.referenceMesh.visible = document.getElementById('show-reference').checked;
            state.scene.add(state.referenceMesh);
        } catch (e) {
            console.warn('Could not load reference cloud');
        }

        console.log('All point clouds loaded');
    } catch (error) {
        console.error('Error loading point clouds:', error);
        alert('Error loading point clouds. Check console for details.');
    } finally {
        loadingEl.classList.add('hidden');
    }
}

// Display a specific segment
function displaySegment(index) {
    // Remove current mesh
    if (state.currentMesh) {
        state.scene.remove(state.currentMesh);
        state.currentMesh = null;
    }

    // Remove change meshes
    if (state.addedMesh) {
        state.scene.remove(state.addedMesh);
        state.addedMesh = null;
    }

    if (state.removedMesh) {
        state.scene.remove(state.removedMesh);
        state.removedMesh = null;
    }

    // Add new mesh
    const geometry = state.pointClouds[index];
    const material = new THREE.PointsMaterial({ size: 0.03, vertexColors: true });
    state.currentMesh = new THREE.Points(geometry, material);
    state.scene.add(state.currentMesh);

    // Add change clouds if enabled
    const showAdded = document.getElementById('show-added').checked;
    const showRemoved = document.getElementById('show-removed').checked;
    const changePair = state.changeCloudsPairs[index];

    if (showAdded && changePair.added) {
        const addedMaterial = new THREE.PointsMaterial({ size: 0.04, color: 0x00ff00 });
        state.addedMesh = new THREE.Points(changePair.added, addedMaterial);
        state.scene.add(state.addedMesh);
    }

    if (showRemoved && changePair.removed) {
        const removedMaterial = new THREE.PointsMaterial({ size: 0.04, color: 0xff0000 });
        state.removedMesh = new THREE.Points(changePair.removed, removedMaterial);
        state.scene.add(state.removedMesh);
    }

    // Update UI
    state.currentSegment = index;
    updateSegmentInfo(index);
}

// Update segment information display
function updateSegmentInfo(index) {
    const segment = state.timeline.segments[index];

    document.getElementById('segment-label').textContent = index;
    document.getElementById('info-segment').textContent = `${index} / ${state.timeline.total_segments - 1}`;
    document.getElementById('info-time').textContent = `${segment.start_time.toFixed(1)}s - ${segment.end_time.toFixed(1)}s`;
    document.getElementById('info-duration').textContent = `${segment.duration_sec.toFixed(1)}s`;

    const pointCount = state.pointClouds[index].attributes.position.count;
    document.getElementById('info-points').textContent = pointCount.toLocaleString();
}

// Setup UI controls
function setupControls() {
    const timeSlider = document.getElementById('time-slider');
    const showReference = document.getElementById('show-reference');
    const showAdded = document.getElementById('show-added');
    const showRemoved = document.getElementById('show-removed');
    const resetCamera = document.getElementById('reset-camera');

    // Setup time slider
    timeSlider.max = state.timeline.total_segments - 1;
    timeSlider.value = 0;

    timeSlider.addEventListener('input', (e) => {
        const index = parseInt(e.target.value);
        displaySegment(index);
    });

    // Reference toggle
    showReference.addEventListener('change', (e) => {
        if (state.referenceMesh) {
            state.referenceMesh.visible = e.target.checked;
        }
    });

    // Change toggles
    showAdded.addEventListener('change', () => {
        displaySegment(state.currentSegment);
    });

    showRemoved.addEventListener('change', () => {
        displaySegment(state.currentSegment);
    });

    // Reset camera
    resetCamera.addEventListener('click', () => {
        state.camera.position.set(5, 5, 5);
        state.controls.reset();
    });

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            const newIndex = Math.max(0, state.currentSegment - 1);
            timeSlider.value = newIndex;
            displaySegment(newIndex);
        } else if (e.key === 'ArrowRight') {
            const newIndex = Math.min(state.timeline.total_segments - 1, state.currentSegment + 1);
            timeSlider.value = newIndex;
            displaySegment(newIndex);
        }
    });
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}

// Initialize application
async function init() {
    console.log('Initializing Temporal Construction World Model viewer...');

    initScene();

    const timeline = await loadTimeline();
    if (!timeline) {
        return;
    }

    await loadAllPointClouds();

    setupControls();

    // Display first segment
    displaySegment(0);

    // Start animation loop
    animate();

    console.log('Viewer initialized successfully');
}

// Start when DOM is ready
init();
