var EventBus = new Vue();
new Vue({
    el: '#app',
    data() {
        return {
            width: 500,
            height: 500,
            data: [
                { x: -1, y: 2 },
                { x: 0, y: 5 },
                { x: 1, y: -4 },
                { x: -2, y: 7 },
                { x: 3, y: 1 }
            ],
            xScale: null,
            yScale: null,
            texts: [],
        };
    },
    created() {
        EventBus.$on('buttonClicked', this.updateData);
    },
    mounted() {
        this.xScale = d3.scaleLinear().range([30, 470]);
        this.yScale = d3.scaleLinear().range([470, 30]);
        this.updateScales();
        this.drawChart();
    },
    methods: {
        sendData() {
            // 发送二维数据到事件总线用于更新第二个散点图
            EventBus.$emit('data-updated', data);
        },
        updateScales() {
            this.xScale.domain([d3.min(this.data, d => d.x), d3.max(this.data, d => d.x)]);
            this.yScale.domain([d3.min(this.data, d => d.y), d3.max(this.data, d => d.y)]);
        },
        drawAxes() {
            const svg = d3.select(this.$refs.svg);

            svg.append("g")
                .attr("transform", "translate(0, 470)")
                .call(d3.axisBottom(this.xScale))
                .append("text")
                .attr("class", "axis-label")
                .attr("x", 250)
                .attr("y", 40)
                .text("X轴");

            svg.append("g")
                .attr("transform", "translate(30, 0)")
                .call(d3.axisLeft(this.yScale))
                .append("text")
                .attr("class", "axis-label")
                .attr("x", -130)
                .attr("y", 150)
                .attr("transform", "rotate(-90)")
                .text("Y轴");
        },
        drawDots() {
            const svg = d3.select(this.$refs.svg);
            const dots = svg.selectAll("circle")
                .data(this.data)
                .join("circle")
                .attr("class", "dot")
                .attr("cx", d => this.xScale(d.x))
                .attr("cy", d => this.yScale(d.y))
                .attr("r", 5);

            dots.on("mouseover", (event, d) => {
                d3.select(event.target)
                    .transition()
                    .attr("fill", "orange")
                    .attr("r", 8);
            });

            dots.on("mouseout", (event, d) => {
                d3.select(event.target)
                    .transition()
                    .attr("fill", "skyblue")
                    .attr("r", 5);
            });

            dots.on("click", (event, d) => {
                console.log("Clicked point:", d);
                EventBus.$emit('data-updated', [d]);
            });
        },
        drawChart() {
            const svg = d3.select(this.$refs.svg);
            svg.selectAll("*").remove();
            this.drawAxes();
            this.drawDots();
        },
        updateData() {
            initialize_overview((data) => {
                this.data = data.map(item => ({ x: item[1][0], y: item[1][1] }));
                this.texts = data.map(item => item[0]);
                this.updateScales();
                this.drawChart();
            });
            this.updateScales();
            this.drawChart();
        },
    },
});
new Vue({
    el: '#app2',
    data() {
        return {
            dataArr: [],
            svg: null,
            data: [],
            timer: null
        };
    },
    mounted() {
        this.svg = d3.select(this.$refs.svg2);
        this.draw();
    },
    created() {
        // 监听事件总线的数据更新事件
        EventBus.$on('data-updated', (data) => {

            if (this.data.length >= 4) {
                this.dataArr.shift();
            }
            // 将接收到的对象压入数组中
            this.dataArr.push(...data);
            console.log(this.dataArr)

            if (this.data.length === 0) {
                this.data.push({ x: 0, y: 0 });
            } else if (this.data.length === 1) {
                this.data.push({ x: 1, y: 0 });
            } else if (this.data.length === 2) {
                this.data.push({ x: 0, y: 1 });
            } else if (this.data.length === 3) {
                this.data.push({ x: 1, y: 1 });
            }
            this.draw();
        });
        EventBus.$on('buttonClicked', () => {
            this.clearDataArr();
            this.clearData();
            this.draw();
        });
    },
    methods: {
        draw(restartAnimation = false) {
            const xScale = d3.scaleLinear()
                .domain([0, d3.max(this.data, d => d.x)])
                .range([30, 270]);

            const yScale = d3.scaleLinear()
                .domain([0, d3.max(this.data, d => d.y)])
                .range([270, 30]);

            this.svg.selectAll("circle").remove();

            const dots = this.svg.selectAll("circle")
                .data(this.data)
                .enter()
                .append("circle")
                .attr("class", "dot")
                .attr("cx", d => xScale(d.x))
                .attr("cy", d => yScale(d.y))
                .attr("r", 15);

            dots.each((d, i, nodes) => {
                this.animate(d3.select(nodes[i]), "dot-" + i);
            });

            dots.on("mouseover", (event, d, i) => {
                // this.showTooltip(event, d); // 显示坐标信息
                d3.select(event.target)
                    .interrupt("dot-" + i)
                    .attr("r", 30)
                    .style("fill", "orange");
            })
            dots.on("mouseout", (event, d, i) => {
                d3.select(event.target)
                    .style("fill", "steelblue");
                if (restartAnimation) {
                    this.animate(d3.select(event.target), "dot-" + i);
                }
                // const tooltip = document.getElementById("tooltip");
                // tooltip.style.visibility = "hidden"; // 隐藏坐标信息
            })
                .on("click", (event, d, i) => {
                    if (!this.clicks) { // 如果没有点击过
                        this.clicks = 1;
                        this.clickTimer = setTimeout(() => {
                            this.clicks = 0;
                        }, 1000);
                    } else {
                        this.clicks++;
                        if (this.clicks === 3) {
                            clearTimeout(this.clickTimer);
                            let pointToDelete;
                            switch (this.data.length) {
                                case 4:
                                    pointToDelete = { x: 1, y: 1 };
                                    break;
                                case 3:
                                    pointToDelete = { x: 0, y: 1 };
                                    break;
                                case 2:
                                    pointToDelete = { x: 1, y: 0 };
                                    break;
                                default:
                                    pointToDelete = { x: 0, y: 0 };
                            }

                            const indexToDelete = this.data.findIndex(item => item.x === pointToDelete.x && item.y === pointToDelete.y);

                            if (indexToDelete !== -1) {
                                this.data.splice(indexToDelete, 1);
                                this.dataArr.splice(indexToDelete, 1); // 删除对应索引处的数据
                                this.svg.selectAll("circle").data(this.data).exit().remove();
                            }
                            this.clicks = 0;
                        }
                    }
                    // console.log(d);
                })

            this.svg.append("g")
                .attr("transform", "translate(0, 270)")
                .call(d3.axisBottom(xScale).tickSize(0).tickFormat(""));

            this.svg.append("g")
                .attr("transform", "translate(30, 0)")
                .call(d3.axisLeft(yScale).tickSize(0).tickFormat(""));

            this.svg.select("#background")
                .attr("x", 30)
                .attr("y", 30)
                .attr("width", 240)
                .attr("height", 240);
        },
        animate(selection, id) {
            selection
                .transition(id)
                .duration(2000)
                .attr("r", 30)
                .style("opacity", 1)
                .transition()
                .duration(2000)
                .attr("r", 15)
                .style("opacity", 1)
                .end().then(() => this.animate(selection, id))
        },
        clearDataArr() {
            this.dataArr = []; // Clear the data array
            console.log("Data Array cleared");
        },
        clearData() {
            this.data = [];
            console.log('data cleared');
        },
        showTooltip(event, d) {
            const tooltip = document.getElementById("tooltip");
            tooltip.style.visibility = "visible";
            tooltip.innerHTML = `X: ${d.x}, Y: ${d.y}`;

            const [x, y] = d3.pointer(event);
            tooltip.style.left = `${x}px`;
            tooltip.style.top = `${y}px`;
        },
    }
});
new Vue({
    el: '#buttonApp',
    methods: {
        updateData() {
            event.preventDefault();
            EventBus.$emit('buttonClicked');
        },
    },
});
