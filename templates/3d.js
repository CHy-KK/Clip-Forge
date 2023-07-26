// Base Settings
const width = 400;
const height = 400;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
const radius = 100;
let dirX = 0;
let dirY = 45;
camera.position.set(Math.sin(dirX) * radius, Math.cos(dirY) * radius, Math.cos(dirX) * radius); // 设置0相机新的位置
camera.lookAt(0, 0, 0);
console.log(camera.position)

const renderer = new THREE.WebGLRenderer();
renderer.setSize(width, height);
renderer.setClearColor(0x808080);
const container = document.getElementById('canvas-container');
container.appendChild(renderer.domElement);

// Light Settings
const light = new THREE.AmbientLight(0xbbbbbb); // soft white light
scene.add(light);
const directionalLightTop = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLightTop.position.z = 1;
scene.add(directionalLightTop);
const directionalLightLeft = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLightLeft.position.z = -2;
scene.add(directionalLightLeft);
const directionalLightRight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLightRight.position.z = 2;
scene.add(directionalLightRight);
const directionalLightDown = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLightRight.position.y = -2;
scene.add(directionalLightDown);

// Create Initial Voxel Cube
const voxelSize = 3;
const geometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
const material = new THREE.MeshLambertMaterial({ color: 0x5588aa });

let cube = new THREE.Mesh(geometry, material);
cube.position.set(0, 0, -10);
scene.add(cube);

let isDragging = false;
let prevX, prevY;
let curX, curY;

// Render loop
function render() {
    requestAnimationFrame(render);
    if (isDragging) {
        dirX = (dirX + (prevX - curX) * 0.01);
        dirY = (dirY + (prevY - curY) * 0.01);
        prevX = curX;
        prevY = curY;
        const x = Math.sin(dirX) * radius;
        const z = Math.cos(dirX) * radius;
        const y = Math.cos(dirY) * radius;
        camera.position.set(x, y, z); // 设置0相机新的位置
        camera.lookAt(0, 0, 0); // 设置相机朝向坐标原点
    }
    renderer.render(scene, camera);
}

render();

function initialize_overview(callback) {
    $.ajax({
        url: '/initialize_overview',
        type: 'POST',
        contentType: 'application/json',
        success: function (data) {
            console.log("success get embedding");
            console.log(data);
            callback(data)
        }
    });
}

function get_embeddings_by_text_query() {
    const mylist = ["plane", "car", "truck", "rocket"];

    $.ajax({
        url: '/get_embeddings_by_text_query',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(mylist),
        success: function (data) {
            console.log("success get embedding");
            console.log(data);

            // [t, p]
            //const coordinates = data;//.map(item => ({ x: item.x, y: item.y }));
            //callback(coordinates);
        }
    });
    event.preventDefault();
}

let voxel_data = []

function update_voxel() {
    let voxel32 = []
    for (let i = 0; i < 64; i += 2) {
        tmp1 = []
        for (let j = 0; j < 64; j += 2) {
            tmp2 = []
            for (let k = 0; k < 64; k += 2) {
                if (Number(voxel_data[i][j][k]) + Number(voxel_data[i + 1][j][k]) + Number(voxel_data[i][j + 1][k]) + Number(voxel_data[i + 1][j + 1][k])
                    + Number(voxel_data[i][j][k + 1]) + Number(voxel_data[i + 1][j][k + 1]) + Number(voxel_data[i][j + 1][k + 1]) + Number(voxel_data[i + 1][j + 1][k + 1]) >= 3)
                    tmp2.push(true)
                else
                    tmp2.push(false)

            }
            tmp1.push(tmp2)
        }
        voxel32.push(tmp1)
    }

    console.log(voxel32)
    $.ajax({
        url: '/update_voxel',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(voxel32),
        success: function (data) {
            console.log("success get embedding");
            console.log(data);
        }
    });
}

function get_voxel(xval, yval, idx0 = 0, idx1 = -1, idx2 = -1, idx3 = -1) {
    $.ajax({
        // 如果值是0那么必须手动打印.0，否则url会自动去掉小数点，然后flask识别时会识别为int，与float类型冲突导致报错
        url: `/get_voxel/${idx0}-${idx1}-${idx2}-${idx3}/${xval}${xval == 0 ? '.0' : ''}-${yval}${yval == 0 ? '.0' : ''}`,
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(''),
        success: function (data) {
            voxel_data = data;
            console.log(scene.children);
            for (let i = 0; i < scene.children.length;) {
                if (scene.children[i] instanceof THREE.Mesh)
                    scene.remove(scene.children[i]);
                else i++;
            }
            let voxels = [];
            for (let x = 0; x < 64; x++) {
                for (let y = 0; y < 64; y++) {
                    for (let z = 0; z < 64; z++) {
                        if (data[z][y][x])
                            voxels.push([x - 32, y - 32, z - 32]);
                    }
                }
            }
            const meshes = new THREE.InstancedMesh(geometry, material, voxels.length);
            const matrix = new THREE.Matrix4();
            for (let i = 0; i < voxels.length; i++) {
                matrix.setPosition(voxels[i][0], voxels[i][1], voxels[i][2]);
                meshes.setMatrixAt(i, matrix);

            }
            scene.add(meshes);
        }
    });

    renderer.domElement.addEventListener('mousedown', function (event) {
        isDragging = true;
        prevX = event.offsetX;
        prevY = event.offsetY;
        curX = event.offsetX;
        curY = event.offsetY;
    })

    renderer.domElement.addEventListener('mousemove', function (event) {
        if (isDragging) {
            curX = event.offsetX;
            curY = event.offsetY;
        }
    })

    renderer.domElement.addEventListener('mouseup', function (event) {
        isDragging = false;
    })
}
