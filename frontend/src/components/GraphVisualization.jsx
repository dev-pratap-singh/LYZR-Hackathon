import { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize2, Search, Minimize2 } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const GraphVisualization = ({ refreshTrigger = 0 }) => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showTable, setShowTable] = useState(false);
  const [linkDistance, setLinkDistance] = useState(350); // Increased default distance for better spacing
  const [chargeStrength, setChargeStrength] = useState(-1500); // Increased repulsion for better spacing
  const [isFullscreen, setIsFullscreen] = useState(false); // Fullscreen mode
  const [isDarkMode, setIsDarkMode] = useState(false); // Track dark mode

  const svgRef = useRef(null);
  const simulationRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  // Detect dark mode changes
  useEffect(() => {
    const checkDarkMode = () => {
      setIsDarkMode(document.body.classList.contains('dark-mode'));
    };

    // Initial check
    checkDarkMode();

    // Watch for dark mode changes
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });

    return () => observer.disconnect();
  }, []);

  // Fetch graph data from backend
  useEffect(() => {
    fetchGraphData();
  }, []);

  // Refetch graph data when refreshTrigger changes (after graph updates)
  useEffect(() => {
    if (refreshTrigger > 0) {
      console.log('Graph update detected, refreshing graph data...');
      fetchGraphData();
    }
  }, [refreshTrigger]);

  // ESC key to exit fullscreen
  useEffect(() => {
    const handleEscKey = (event) => {
      if (event.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
        setSelectedNode(null);
        setSelectedEdge(null);
      }
    };

    window.addEventListener('keydown', handleEscKey);
    return () => window.removeEventListener('keydown', handleEscKey);
  }, [isFullscreen]);

  const fetchGraphData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/api/rag/graph/data`);

      if (response.data.nodes.length === 0) {
        setError('No graph data found. Please process the document with GraphRAG first.');
        setLoading(false);
        return;
      }

      setNodes(response.data.nodes);
      setEdges(response.data.edges);
      setStats(response.data.stats);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching graph data:', err);
      setError(err.response?.data?.detail || 'Failed to load graph data');
      setLoading(false);
    }
  };

  // D3 Visualization
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    // Dynamic sizing based on fullscreen mode
    const width = isFullscreen ? window.innerWidth - 40 : 1000;
    const height = isFullscreen ? window.innerHeight - 200 : 600;

    // Clear previous content
    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Zoom behavior - increased extent for better zoom capability
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);
    zoomBehaviorRef.current = zoom;

    // Calculate node radius based on text length (subtle scaling)
    const getNodeRadius = (node) => {
      const label = node.label || node.id;
      const textLength = label.length;
      // Base radius: 50, grows slightly with text length
      // Max radius: 70 for very long text
      return Math.min(50 + (textLength * 0.8), 70);
    };

    // Store radius on each node for later use
    nodes.forEach(node => {
      node.radius = getNodeRadius(node);
    });

    // Create node ID set for fast lookup
    const nodeIds = new Set(nodes.map(n => n.id));

    // Filter out edges with missing source or target nodes
    const validEdges = edges.filter(e => {
      const hasSource = nodeIds.has(e.source);
      const hasTarget = nodeIds.has(e.target);

      if (!hasSource || !hasTarget) {
        console.warn(`⚠️ Skipping orphaned edge: ${e.source} -> ${e.target} (${e.type})`);
        if (!hasSource) console.warn(`  Missing source node: ${e.source}`);
        if (!hasTarget) console.warn(`  Missing target node: ${e.target}`);
      }

      return hasSource && hasTarget;
    });

    // Log summary if edges were filtered
    if (validEdges.length < edges.length) {
      console.warn(`⚠️ Graph Data Integrity Issue: ${edges.length - validEdges.length} orphaned edges filtered out`);
      console.warn('💡 Tip: These edges reference nodes that don\'t exist. Consider cleaning up your graph database.');
    }

    // Create force simulation - adjusted for dynamic node sizes and spacing
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(validEdges.map(e => ({
        source: nodes.find(n => n.id === e.source),
        target: nodes.find(n => n.id === e.target),
        ...e
      }))).id(d => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => d.radius + 15)); // Increased collision padding

    simulationRef.current = simulation;

    // Define arrow markers for different edge types
    const defs = svg.append('defs');

    // Get unique edge types for different colored arrows
    const edgeTypes = [...new Set(validEdges.map(e => e.type))];
    edgeTypes.forEach((type, index) => {
      const color = d3.schemeCategory10[index % 10];
      defs.append('marker')
        .attr('id', `arrowhead-${type}`)
        .attr('viewBox', '-10 -5 10 10')
        .attr('refX', 65)  // Adjusted for slightly larger nodes
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M-10,-5 L0,0 L-10,5')
        .attr('fill', color);
    });

    // Draw edges (using validEdges to exclude orphaned relationships)
    const link = g.append('g')
      .selectAll('g')
      .data(validEdges)
      .join('g');

    const linkLine = link.append('line')
      .attr('stroke', (d, i) => d3.schemeCategory10[edgeTypes.indexOf(d.type) % 10])
      .attr('stroke-width', 2)
      .attr('marker-end', d => `url(#arrowhead-${d.type})`);

    const linkLabel = link.append('text')
      .attr('text-anchor', 'middle')
      .attr('fill', isDarkMode ? '#b0b0b0' : '#475569')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('dy', -5)
      .text(d => d.label)
      .style('cursor', 'pointer')
      .style('pointer-events', 'all')
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedEdge(d);
        setSelectedNode(null);
      });

    // Get unique node types for color coding
    const nodeTypes = [...new Set(nodes.map(n => n.type))];
    const colorScale = d3.scaleOrdinal()
      .domain(nodeTypes)
      .range(d3.schemeCategory10);

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', d => d.radius)  // Use dynamic radius based on text length
      .attr('fill', d => colorScale(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 3)
      .style('cursor', 'pointer');

    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', 'white')
      .attr('font-weight', 'bold')
      .attr('font-size', '14px')
      .text(d => {
        // Show longer labels with better truncation, adjusted for node size
        const label = d.label || d.id;
        // Larger nodes can show more text
        const maxChars = d.radius > 60 ? 30 : 25;
        return label.length > maxChars ? label.substring(0, maxChars - 3) + '...' : label;
      })
      .style('pointer-events', 'none');

    node.on('click', (event, d) => {
      event.stopPropagation();
      setSelectedNode(d);
      setSelectedEdge(null);
    });

    // Add hover effect - scales proportionally to base size
    node.on('mouseover', function(_event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', d.radius + 10);  // Grows by 10px on hover
    })
    .on('mouseout', function(_event, d) {
      d3.select(this).select('circle')
        .transition()
        .duration(200)
        .attr('r', d.radius);  // Back to original size
    });

    // Update positions on tick
    simulation.on('tick', () => {
      linkLine
        .attr('x1', d => {
          const source = nodes.find(n => n.id === d.source);
          return source ? source.x : 0;
        })
        .attr('y1', d => {
          const source = nodes.find(n => n.id === d.source);
          return source ? source.y : 0;
        })
        .attr('x2', d => {
          const target = nodes.find(n => n.id === d.target);
          return target ? target.x : 0;
        })
        .attr('y2', d => {
          const target = nodes.find(n => n.id === d.target);
          return target ? target.y : 0;
        });

      linkLabel
        .attr('x', d => {
          const source = nodes.find(n => n.id === d.source);
          const target = nodes.find(n => n.id === d.target);
          return source && target ? (source.x + target.x) / 2 : 0;
        })
        .attr('y', d => {
          const source = nodes.find(n => n.id === d.source);
          const target = nodes.find(n => n.id === d.target);
          return source && target ? (source.y + target.y) / 2 : 0;
        });

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Clear selection on background click
    svg.on('click', () => {
      setSelectedNode(null);
      setSelectedEdge(null);
    });

    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [nodes, edges, linkDistance, chargeStrength, isFullscreen, isDarkMode]);

  const handleZoomIn = () => {
    if (!zoomBehaviorRef.current || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 1.5);
  };

  const handleZoomOut = () => {
    if (!zoomBehaviorRef.current || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(300).call(zoomBehaviorRef.current.scaleBy, 0.67);
  };

  const handleResetZoom = () => {
    if (!zoomBehaviorRef.current || !svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(500).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };

  const handleClearGraph = async () => {
    if (!window.confirm('Are you sure you want to clear all graph data? This will delete all entities and relationships from the knowledge graph. You can rebuild it by asking a new question.')) {
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/api/rag/graph/clear`);

      if (response.data.success) {
        alert(`✅ ${response.data.message}\n\nThe graph will be rebuilt automatically on your next query.`);
        // Refresh the graph data
        fetchGraphData();
      }
    } catch (err) {
      console.error('Error clearing graph:', err);
      alert('❌ Failed to clear graph data. Please try again.');
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await axios.get(`${API_URL}/api/rag/graph/search`, {
        params: {
          query: searchQuery
        }
      });

      if (response.data.entities.length > 0) {
        // Highlight found entities
        const foundIds = response.data.entities.map(e => e.id);

        // Update node selection
        const foundNode = nodes.find(n => foundIds.includes(n.id));
        if (foundNode) {
          setSelectedNode(foundNode);
        }
      } else {
        alert('No entities found matching your search');
      }
    } catch (err) {
      console.error('Search error:', err);
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
    // Reset selections when toggling fullscreen
    setSelectedNode(null);
    setSelectedEdge(null);
  };

  if (loading) {
    return (
      <div className="graph-loading">
        <div className="loading-spinner">⏳</div>
        <p>Loading graph data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="graph-error">
        <p>⚠️ {error}</p>
        <button onClick={fetchGraphData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  if (nodes.length === 0) {
    return (
      <div className="graph-empty">
        <p>No graph data available for this document.</p>
        <p>Please process the document with GraphRAG first.</p>
      </div>
    );
  }

  return (
    <div className={`graph-visualization-container ${isFullscreen ? 'fullscreen' : ''}`}>
      <div className="graph-header">
        <h2>🕸️ Knowledge Graph Visualization</h2>
        {stats && (
          <div className="graph-stats">
            <span className="stat">
              <strong>{stats.node_count}</strong> Nodes
            </span>
            <span className="stat">
              <strong>{stats.edge_count}</strong> Edges
            </span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="graph-controls">
        <div className="search-controls">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search entities..."
            className="search-input"
          />
          <button onClick={handleSearch} className="search-button">
            <Search size={18} />
          </button>
        </div>

        {/* Spacing Controls */}
        <div className="spacing-controls">
          <div className="slider-control">
            <label htmlFor="link-distance">
              Node Distance: <strong>{linkDistance}</strong>
            </label>
            <input
              id="link-distance"
              type="range"
              min="100"
              max="600"
              step="50"
              value={linkDistance}
              onChange={(e) => setLinkDistance(Number(e.target.value))}
              className="distance-slider"
            />
          </div>
          <div className="slider-control">
            <label htmlFor="charge-strength">
              Repulsion: <strong>{Math.abs(chargeStrength)}</strong>
            </label>
            <input
              id="charge-strength"
              type="range"
              min="-3000"
              max="-500"
              step="100"
              value={chargeStrength}
              onChange={(e) => setChargeStrength(Number(e.target.value))}
              className="distance-slider"
            />
          </div>
        </div>

        <div className="zoom-controls">
          <button onClick={handleZoomIn} title="Zoom In" className="control-button">
            <ZoomIn size={18} />
          </button>
          <button onClick={handleZoomOut} title="Zoom Out" className="control-button">
            <ZoomOut size={18} />
          </button>
          <button onClick={handleResetZoom} title="Reset Zoom" className="control-button">
            <Maximize2 size={18} />
          </button>
          <button
            onClick={toggleFullscreen}
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            className="control-button fullscreen-button"
          >
            {isFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            <span style={{ marginLeft: '6px' }}>{isFullscreen ? 'Exit' : 'Fullscreen'}</span>
          </button>
          <button
            onClick={() => setShowTable(!showTable)}
            className="control-button"
          >
            {showTable ? 'Hide' : 'Show'} Table
          </button>
          <button
            onClick={handleClearGraph}
            title="Clear All Graph Data"
            className="control-button clear-graph-button"
          >
            🗑️ Clear Graph
          </button>
        </div>
      </div>

      {/* Graph Canvas */}
      <div className="graph-canvas-wrapper">
        <svg
          ref={svgRef}
          width={isFullscreen ? window.innerWidth - 40 : 1000}
          height={isFullscreen ? window.innerHeight - 200 : 600}
          className="graph-canvas"
        />

        {/* Details Panel */}
        {(selectedNode || selectedEdge) && (
          <div className="details-panel">
            {selectedNode && (
              <div className="node-details">
                <div className="details-header">
                  <h3>📍 Node Details</h3>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="close-button"
                  >
                    ×
                  </button>
                </div>
                <div className="details-content">
                  <p><strong>ID:</strong> {selectedNode.id}</p>
                  <p><strong>Type:</strong> {selectedNode.type}</p>
                  {selectedNode.description && (
                    <p><strong>Description:</strong> {selectedNode.description}</p>
                  )}
                  {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                    <div>
                      <strong>Properties:</strong>
                      <pre>{JSON.stringify(selectedNode.properties, null, 2)}</pre>
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedEdge && (
              <div className="edge-details">
                <div className="details-header">
                  <h3>🔗 Relationship Details</h3>
                  <button
                    onClick={() => setSelectedEdge(null)}
                    className="close-button"
                  >
                    ×
                  </button>
                </div>
                <div className="details-content">
                  <p><strong>Type:</strong> {selectedEdge.type}</p>
                  <p><strong>Source:</strong> {selectedEdge.source}</p>
                  <p><strong>Target:</strong> {selectedEdge.target}</p>
                  {selectedEdge.properties && Object.keys(selectedEdge.properties).length > 0 && (
                    <div>
                      <strong>Properties:</strong>
                      <pre>{JSON.stringify(selectedEdge.properties, null, 2)}</pre>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Data Table */}
      {showTable && (
        <div className="graph-table">
          <div className="table-section">
            <h3>Entities ({nodes.length})</h3>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Connections</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map(node => (
                    <tr
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      className={selectedNode?.id === node.id ? 'selected' : ''}
                    >
                      <td>{node.id}</td>
                      <td>{node.type}</td>
                      <td>{node.description || '-'}</td>
                      <td>
                        {edges.filter(e => e.source === node.id || e.target === node.id).length}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="table-section">
            <h3>Relationships ({edges.length})</h3>
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>Relationship</th>
                    <th>Target</th>
                  </tr>
                </thead>
                <tbody>
                  {edges.map(edge => (
                    <tr
                      key={edge.id}
                      onClick={() => setSelectedEdge(edge)}
                      className={selectedEdge?.id === edge.id ? 'selected' : ''}
                    >
                      <td>{edge.source}</td>
                      <td className="relationship-type">{edge.type}</td>
                      <td>{edge.target}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphVisualization;
