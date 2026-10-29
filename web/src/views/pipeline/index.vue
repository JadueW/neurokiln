<template>
  <div class="app-container">
    <div class="page-head">
      <div>
        <h2 class="page-title">{{ $t('menu.pipeline') }}</h2>
        <p class="page-desc">{{ $t('pipeline.pageDesc') }}</p>
      </div>
      <div class="head-actions">
        <el-button icon="Shop" @click="openMarket">{{ $t('templates.title') }}</el-button>
        <el-button icon="Refresh" @click="newGraph">{{ $t('pipeline.newGraph') }}</el-button>
        <el-button icon="Rank" @click="autoLayout">{{ $t('pipeline.autoLayout') }}</el-button>
        <el-button icon="CircleCheck" :loading="validating" @click="validateGraph">{{ $t('pipeline.validate') }}</el-button>
        <el-button type="primary" icon="Upload" :loading="saving" @click="saveGraph">{{ $t('common.save') }}</el-button>
        <el-button icon="Operation" :disabled="!graph.id" @click="openSpace">{{ $t('pipeline.searchSpace') }}</el-button>
      </div>
    </div>

    <el-card shadow="never" body-style="padding:12px 16px" style="margin-bottom:12px">
      <div class="toolbar">
        <span class="toolbar-label">{{ $t('pipeline.graphName') }}</span>
        <el-input v-model="graph.name" style="width:200px" maxlength="80" :placeholder="$t('pipeline.graphNamePlaceholder')" />
        <span class="toolbar-label">{{ $t('pipeline.templateLabel') }}</span>
        <el-select v-model="templateKey" style="width:220px" :placeholder="$t('pipeline.templatePlaceholder')" @change="applyTemplate">
          <el-option v-for="item in TEMPLATES" :key="item.key" :label="$t(item.label)" :value="item.key" />
        </el-select>
        <span class="toolbar-label">{{ $t('pipeline.myPipelines') }}</span>
        <el-select
          v-model="loadedId"
          style="width:220px"
          clearable
          :placeholder="$t('pipeline.loadPipelinePlaceholder')"
          @change="loadCustom"
        >
          <el-option
            v-for="item in customs"
            :key="item.id"
            :label="`${item.name} · #${item.id}`"
            :value="item.id"
          />
        </el-select>
        <el-button type="danger" plain icon="Delete" :disabled="!graph.id" @click="removeCurrent">{{ $t('pipeline.deleteCurrent') }}</el-button>
        <el-button icon="Document" :disabled="!graph.id" @click="exportCurrent">{{ $t('pipeline.exportTemplate') }}</el-button>
        <el-button icon="FolderOpened" @click="triggerTemplateImport">{{ $t('pipeline.importTemplate') }}</el-button>
        <input
          ref="templateFileInput"
          type="file"
          accept=".json,application/json"
          style="display:none"
          @change="onTemplateFile"
        />
        <el-tag v-if="graph.id" type="success" effect="plain">{{ $t('pipeline.savedTag', { id: graph.id }) }}</el-tag>
        <el-tag v-else type="info" effect="plain">{{ $t('pipeline.unsavedTag') }}</el-tag>
      </div>
    </el-card>

    <el-card shadow="never" body-style="padding:12px 16px" style="margin-bottom:12px">
      <div class="palette-head">
        <span class="palette-heading">{{ $t('pipeline.nodeLibrary') }}</span>
        <el-input
          v-model="nodeQuery"
          size="small"
          clearable
          :placeholder="$t('pipeline.nodeSearchPlaceholder')"
          prefix-icon="Search"
          style="width:260px"
        />
        <span class="text-muted" style="font-size:12px">
          {{ $t('pipeline.nodeCount', { filtered: filteredCount, total: catalog.length }) }}
        </span>
      </div>
      <div class="palette">
        <div v-for="group in paletteGroups" :key="group.kind" class="palette-group">
          <span class="palette-kind" :style="{ color: group.color }">{{ group.label }}</span>
          <el-button
            v-for="item in group.items"
            :key="item.type"
            size="small"
            plain
            class="palette-chip"
            :title="tr(item.description)"
            @click="addNode(item)"
          >
            <span class="palette-dot" :style="{ background: group.color }" />{{ tr(item.label) }}
            <span v-if="item.plugin" class="palette-plugin">{{ $t('pipeline.pluginBadge') }}</span>
          </el-button>
          <span v-if="!group.items.length" class="text-muted" style="font-size:12px">{{ $t('pipeline.noMatchedNodes') }}</span>
        </div>
      </div>
    </el-card>

    <el-row :gutter="12">
      <el-col :xs="24" :lg="18" class="editor-col">
        <el-card shadow="never" body-style="padding:0" class="canvas-card">
          <template #header>
            <div class="canvas-head">
              <span><b>{{ $t('pipeline.canvas') }}</b> · {{ $t('pipeline.canvasStats', { nodes: graph.nodes.length, edges: graph.edges.length }) }}</span>
              <span class="text-muted hint">
                {{ pendingSource
                  ? $t('pipeline.hintPending', { source: pendingSource, port: pendingPort ? ` · ${pendingPort}` : '' })
                  : $t('pipeline.hintIdle') }}
              </span>
            </div>
          </template>
          <div ref="viewportEl" class="canvas-viewport">
            <div
              ref="surfaceEl"
              class="canvas-surface"
              :style="{ width: `${canvasSize.width}px`, height: `${canvasSize.height}px` }"
              @click.self="clearSelection"
            >
              <svg class="edges" :width="canvasSize.width" :height="canvasSize.height">
                <defs>
                  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#a8b3c4" />
                  </marker>
                  <marker id="arrow-selected" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#409eff" />
                  </marker>
                </defs>
                <g v-for="(edge, index) in graph.edges" :key="edgeKey(edge)">
                  <path class="edge-hit" :d="edgePath(edge)" @click.stop="selectEdge(index)" />
                  <path
                    class="edge-line"
                    :class="{ selected: selectedEdgeIndex === index }"
                    :d="edgePath(edge)"
                    :marker-end="selectedEdgeIndex === index ? 'url(#arrow-selected)' : 'url(#arrow)'"
                  />
                  <text v-if="edge.source_port" class="edge-label" :x="edgeMid(edge).x - 18" :y="edgeMid(edge).y - 6">
                    {{ edgePortLabel(edge) }}
                  </text>
                </g>
              </svg>

              <div
                v-for="node in graph.nodes"
                :key="node.id"
                class="node"
                :class="{ selected: selectedNodeId === node.id, pending: pendingSource === node.id }"
                :style="nodeStyle(node)"
                @click.stop="selectNode(node.id)"
              >
                <div class="node-head" @pointerdown.stop.prevent="startDrag($event, node)">
                  <span class="node-title">{{ tr(node.label) }}</span>
                  <span class="node-kind" :style="{ color: colorFor(node), background: `${colorFor(node)}1a` }">
                    {{ kindLabel(node) }}
                  </span>
                </div>
                <div class="node-body">
                  <div class="mono node-type">{{ node.type }}</div>
                  <div class="node-summary">{{ summary(node) || $t('pipeline.noParams') }}</div>
                </div>
                <button class="port port-in" :title="$t('pipeline.portInput')" @click.stop="connectTo(node.id)" />
                <span v-if="portsFor(node).length" class="port-anchor" @click.stop>
                  <el-dropdown trigger="click" placement="right-start" @command="(port) => beginConnect(node.id, port)">
                    <button
                      class="port port-out port-multi"
                      :class="{ active: pendingSource === node.id }"
                      :title="$t('pipeline.selectSubBand', { count: portsFor(node).length })"
                    />
                    <template #dropdown>
                      <el-dropdown-menu>
                        <el-dropdown-item v-for="port in portsFor(node)" :key="port.name" :command="port.name">
                          <span class="mono">{{ port.name }}</span>
                          <span class="text-muted" style="margin-left:6px">{{ port.label }}</span>
                        </el-dropdown-item>
                      </el-dropdown-menu>
                    </template>
                  </el-dropdown>
                </span>
                <button
                  v-else-if="canOutput(node)"
                  class="port port-out"
                  :class="{ active: pendingSource === node.id }"
                  :title="$t('pipeline.portOutput')"
                  @click.stop="beginConnect(node.id)"
                />
              </div>

              <div v-if="!graph.nodes.length" class="empty-canvas">
                {{ $t('pipeline.emptyCanvas') }}
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="6" class="editor-col">
        <el-card shadow="never" class="inspector">
          <template #header>
            <b>{{ selectedNode ? $t('pipeline.nodeSettings') : selectedEdge ? $t('pipeline.edgeSettings') : $t('pipeline.graphSettings') }}</b>
          </template>

          <template v-if="selectedNode && selectedSpec">
            <el-form label-position="top" size="default">
              <el-form-item :label="$t('pipeline.displayName')">
                <el-input v-model="selectedNode.label" maxlength="80" />
              </el-form-item>
              <el-form-item :label="$t('pipeline.nodeType')">
                <span class="mono node-type-text">{{ selectedNode.type }}</span>
              </el-form-item>
            </el-form>
            <el-alert :title="tr(selectedSpec.description)" type="info" :closable="false" class="node-desc" />
            <div class="node-meta">
              <el-tag v-if="selectedSpec.plugin" size="small" type="danger" effect="plain">{{ $t('pipeline.pluginNode') }}</el-tag>
              <el-tag size="small" type="info" effect="plain">{{ $t('pipeline.paradigmBadge', { paradigm: selectedSpec.paradigm }) }}</el-tag>
              <el-tag size="small" type="info" effect="plain">
                {{ (selectedSpec.inputs || []).join(' / ') }} → {{ (selectedSpec.outputs || []).join(' / ') }}
              </el-tag>
              <el-tag v-if="selectedSpec.requires_fit" size="small" type="warning" effect="plain">{{ $t('pipeline.requiresFit') }}</el-tag>
              <el-tag v-if="selectedSpec.cacheable" size="small" type="success" effect="plain">{{ $t('pipeline.cacheable') }}</el-tag>
            </div>
            <el-form v-if="paramFields.length" label-position="top" style="margin-top:12px">
              <el-form-item v-for="field in paramFields" :key="field.name" :label="tr(field.label)">
                <el-switch
                  v-if="field.type === 'boolean'"
                  v-model="selectedNode.params[field.name]"
                  inline-prompt
                  :active-text="$t('pipeline.on')"
                  :inactive-text="$t('pipeline.off')"
                />
                <el-input-number
                  v-else-if="field.type === 'integer' || field.type === 'number'"
                  v-model="selectedNode.params[field.name]"
                  :min="field.schema.min"
                  :max="field.schema.max"
                  :step="field.schema.step || (field.type === 'integer' ? 1 : 0.1)"
                  :precision="field.type === 'integer' ? 0 : precisionFor(field.schema)"
                  style="width:100%"
                />
                <el-select
                  v-else-if="field.schema.enum"
                  v-model="selectedNode.params[field.name]"
                  style="width:100%"
                >
                  <el-option
                    v-for="option in field.schema.enum"
                    :key="String(option)"
                    :label="String(option)"
                    :value="option"
                  />
                </el-select>
                <el-input v-else v-model="selectedNode.params[field.name]" />
              </el-form-item>
              <el-button link type="primary" size="small" icon="RefreshLeft" @click="resetParams">
                {{ $t('pipeline.restoreDefaults') }}
              </el-button>
            </el-form>
            <div class="inspector-actions">
              <el-button type="danger" plain icon="Delete" @click="removeSelectedNode">{{ $t('pipeline.deleteNode') }}</el-button>
            </div>
          </template>

          <template v-else-if="selectedEdge">
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item :label="$t('pipeline.upstream')">{{ nodeLabel(selectedEdge.source) }}</el-descriptions-item>
              <el-descriptions-item :label="$t('pipeline.downstream')">{{ nodeLabel(selectedEdge.target) }}</el-descriptions-item>
              <el-descriptions-item :label="$t('pipeline.connection')">
                <span class="mono" style="font-size:12px">{{ selectedEdge.source }} → {{ selectedEdge.target }}</span>
              </el-descriptions-item>
              <el-descriptions-item v-if="edgePorts(selectedEdge).length" :label="$t('pipeline.outputSubBand')">
                <el-select
                  v-model="selectedEdge.source_port"
                  size="small"
                  style="width:100%"
                  @change="validation = null"
                >
                  <el-option
                    v-for="port in edgePorts(selectedEdge)"
                    :key="port.name"
                    :label="`${port.name} · ${port.label}`"
                    :value="port.name"
                  />
                </el-select>
              </el-descriptions-item>
            </el-descriptions>
            <div class="inspector-actions">
              <el-button type="danger" plain icon="Delete" @click="removeSelectedEdge">{{ $t('pipeline.deleteEdge') }}</el-button>
            </div>
          </template>

          <template v-else>
            <el-form label-position="top">
              <el-form-item :label="$t('pipeline.description')">
                <el-input
                  v-model="graph.description"
                  type="textarea"
                  :rows="3"
                  maxlength="500"
                  :placeholder="$t('pipeline.descriptionPlaceholder')"
                />
              </el-form-item>
            </el-form>
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item :label="$t('pipeline.nodesLabel')">{{ graph.nodes.length }}</el-descriptions-item>
              <el-descriptions-item :label="$t('pipeline.edgesLabel')">{{ graph.edges.length }}</el-descriptions-item>
              <el-descriptions-item :label="$t('pipeline.classifier')" :span="2">{{ decoderLabel || '-' }}</el-descriptions-item>
            </el-descriptions>
            <ol class="guard-list" style="margin-top:12px">
              <li>{{ $t('pipeline.guardTransfer') }}</li>
              <li>{{ $t('pipeline.guardFusion') }}</li>
              <li>{{ $t('pipeline.guardPorts') }}</li>
              <li>{{ $t('pipeline.guardFit') }}</li>
            </ol>
          </template>

          <el-divider style="margin:16px 0" />
          <div class="validation-head">
            <b>{{ $t('pipeline.validationTitle') }}</b>
            <el-tag v-if="validation" size="small" :type="validation.valid ? 'success' : 'danger'">
              {{ validation.valid ? $t('pipeline.validationPassed') : $t('pipeline.validationFailed') }}
            </el-tag>
            <el-tag v-else size="small" type="info">{{ $t('pipeline.notValidated') }}</el-tag>
          </div>
          <el-alert
            v-if="validation && !validation.valid"
            type="error"
            :closable="false"
            :title="validation.error"
            style="margin-top:8px"
          />
          <div v-else-if="validation && validation.valid" class="validation-body">
            <p>{{ $t('pipeline.topology', { topology: topologyText }) }}</p>
            <p class="text-muted">{{ $t('pipeline.validationCounts', { nodes: validation.nodes, edges: validation.edges }) }}</p>
            <p class="text-muted">
              {{ $t('pipeline.validationDecoder', { decoders: validationDecoderLabel, output: nodeLabel(validation.prediction || validation.decoder) }) }}
            </p>
          </div>
          <p v-else class="text-muted validation-body">
            {{ $t('pipeline.validationHint') }}
          </p>
        </el-card>
      </el-col>
    </el-row>

    <el-dialog v-model="spaceDialog" :title="$t('pipeline.searchSpace')" width="880px" append-to-body>
      <el-alert
        type="info"
        :closable="false"
        show-icon
        :title="$t('pipeline.spaceIntro')"
        style="margin-bottom:12px"
      />
      <div class="space-meta">
        <span class="text-muted">{{ $t('pipeline.spaceHint') }}</span>
        <el-tag :type="spaceEnabledCount > 16 ? 'danger' : 'info'" effect="plain">
          {{ $t('pipeline.spaceDimCount', { count: spaceEnabledCount }) }}
        </el-tag>
      </div>
      <el-table
        v-loading="spaceLoading"
        :data="spaceRows"
        size="small"
        border
        max-height="440"
        row-key="key"
      >
        <el-table-column :label="$t('pipeline.columnEnabled')" width="60" align="center">
          <template #default="{ row }">
            <el-switch v-model="row.enabled" size="small" />
          </template>
        </el-table-column>
        <el-table-column :label="$t('pipeline.columnNode')" min-width="150">
          <template #default="{ row }">
            <div class="space-node">{{ row.node_label }}</div>
            <div class="mono space-type">{{ row.node_type }}</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('pipeline.columnParam')" min-width="120">
          <template #default="{ row }">
            <div>{{ tr(row.label) }}</div>
            <div class="mono space-type">{{ row.param }}</div>
          </template>
        </el-table-column>
        <el-table-column :label="$t('pipeline.columnType')" width="70" align="center">
          <template #default="{ row }">
            <el-tag size="small" effect="plain">{{ spaceKindLabel(row.kind) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="$t('pipeline.columnCurrent')" width="84" align="center">
          <template #default="{ row }"><span class="mono">{{ row.current }}</span></template>
        </el-table-column>
        <el-table-column :label="$t('pipeline.columnRange')" min-width="300">
          <template #default="{ row }">
            <el-select
              v-if="row.kind === 'select'"
              v-model="row.selectedOptions"
              multiple
              size="small"
              :disabled="!row.enabled"
              style="width:100%"
            >
              <el-option v-for="option in row.options" :key="option" :label="option" :value="option" />
            </el-select>
            <div v-else class="space-range">
              <el-input-number
                v-model="row.low"
                class="space-num"
                size="small"
                :controls="false"
                :min="row.schemaMin"
                :max="row.schemaMax"
                :step="row.step || (row.kind === 'int' ? 1 : 0.1)"
                :precision="row.kind === 'int' ? 0 : undefined"
                :disabled="!row.enabled"
                :placeholder="$t('pipeline.lowerBound')"
              />
              <span class="space-dash">–</span>
              <el-input-number
                v-model="row.high"
                class="space-num"
                size="small"
                :controls="false"
                :min="row.schemaMin"
                :max="row.schemaMax"
                :step="row.step || (row.kind === 'int' ? 1 : 0.1)"
                :precision="row.kind === 'int' ? 0 : undefined"
                :disabled="!row.enabled"
                :placeholder="$t('pipeline.upperBound')"
              />
              <span class="space-step-label">{{ $t('pipeline.stepLabel') }}</span>
              <el-input-number
                v-model="row.step"
                class="space-num"
                size="small"
                :controls="false"
                :min="0"
                :step="row.kind === 'int' ? 1 : 0.1"
                :precision="row.kind === 'int' ? 0 : undefined"
                :disabled="!row.enabled"
                :placeholder="$t('pipeline.stepOptional')"
              />
            </div>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty :description="$t('pipeline.spaceEmpty')" :image-size="72" />
        </template>
      </el-table>
      <template #footer>
        <div class="space-footer">
          <span class="text-muted">{{ $t('pipeline.spaceSavedDims', { count: spaceTotal }) }}</span>
          <div>
            <el-button @click="spaceDialog = false">{{ $t('common.cancel') }}</el-button>
            <el-button :disabled="!spaceTotal" :loading="spaceSaving" @click="clearSpace">{{ $t('pipeline.clear') }}</el-button>
            <el-button
              :disabled="!spaceEnabledCount || spaceEnabledCount > 16"
              :loading="spaceSaving"
              @click="saveSpace(false)"
            >{{ $t('common.save') }}</el-button>
            <el-button
              type="primary"
              :disabled="!spaceEnabledCount || spaceEnabledCount > 16"
              :loading="spaceSaving"
              @click="saveSpace(true)"
            >{{ $t('pipeline.saveAndTune') }}</el-button>
          </div>
        </div>
      </template>
    </el-dialog>

    <el-dialog v-model="marketVisible" :title="$t('templates.title')" width="880px">
      <el-tabs v-model="marketTab">
        <el-tab-pane :label="$t('templates.title')" name="templates">
          <p class="text-muted market-desc">{{ $t('templates.description') }}</p>
          <div v-loading="marketLoading" class="market-grid">
            <el-card v-for="item in marketTemplates" :key="item.id" shadow="hover" class="market-card">
              <div class="market-head">
                <b>{{ item.name }}</b>
                <el-tag size="small" effect="plain">{{ item.paradigm }}</el-tag>
              </div>
              <p class="market-text">{{ tr(item.description) }}</p>
              <div class="market-tags">
                <el-tag size="small" type="info" effect="plain">
                  {{ $t('templates.nodes', { count: item.node_count ?? item.spec?.nodes?.length ?? 0 }) }}
                </el-tag>
                <el-tag size="small" :type="item.search_space_count ? 'success' : 'info'" effect="plain">
                  {{ item.search_space_count
                    ? $t('templates.searchDims', { count: item.search_space_count })
                    : $t('templates.noSpace') }}
                </el-tag>
                <el-tag v-for="tag in item.tags || []" :key="tag" size="small" effect="plain">{{ tr(tag) }}</el-tag>
              </div>
              <div v-if="item.paper?.title" class="market-paper">
                <span class="market-label">{{ $t('templates.paper') }}</span>
                {{ item.paper.authors }} ({{ item.paper.year }}). {{ item.paper.title }}
                <span v-if="item.paper.doi" class="mono"> doi:{{ item.paper.doi }}</span>
              </div>
              <div v-if="item.reproduction?.caveats" class="market-paper">
                <span class="market-label">{{ $t('templates.caveats') }}</span>{{ tr(item.reproduction.caveats) }}
              </div>
              <div class="market-actions">
                <el-button size="small" @click="loadTemplateToCanvas(item)">{{ $t('templates.loadCanvas') }}</el-button>
                <el-button
                  size="small"
                  type="primary"
                  :loading="marketInstalling === item.id"
                  @click="installTemplate(item)"
                >
                  {{ $t('templates.install') }}
                </el-button>
              </div>
            </el-card>
            <el-empty v-if="!marketLoading && !marketTemplates.length" :description="$t('pipeline.noTemplates')" />
          </div>
        </el-tab-pane>

        <el-tab-pane :label="$t('templates.reproductionTab')" name="reproduction">
          <p class="text-muted market-desc">{{ $t('templates.reproductionHint') }}</p>
          <div v-loading="reproductionLoading" class="market-grid">
            <el-card v-for="entry in reproductionEntries" :key="entry.id" shadow="hover" class="market-card">
              <div class="market-head">
                <b>{{ entry.name }}</b>
                <el-tag size="small" effect="plain">{{ entry.paradigm }}</el-tag>
              </div>
              <p class="market-text">{{ tr(entry.description) }}</p>
              <div v-if="entry.paper?.title" class="market-paper">
                <span class="market-label">{{ $t('templates.paper') }}</span>
                {{ entry.paper.authors }} ({{ entry.paper.year }}). {{ entry.paper.title }}
                <span v-if="entry.paper.doi" class="mono"> doi:{{ entry.paper.doi }}</span>
              </div>
              <div class="market-paper">
                <span class="market-label">{{ $t('templates.protocol') }}</span>{{ tr(entry.protocol) }}
              </div>
              <div class="market-paper">
                <span class="market-label">{{ $t('templates.datasets') }}</span>{{ (entry.datasets || []).join(' / ') }}
              </div>
              <div class="market-paper">
                <span class="market-label">{{ $t('templates.caveats') }}</span>{{ tr(entry.caveats) }}
              </div>
              <div class="market-actions">
                <el-button size="small" @click="loadReproductionToCanvas(entry)">{{ $t('templates.loadCanvas') }}</el-button>
              </div>
            </el-card>
            <el-empty v-if="!reproductionLoading && !reproductionEntries.length" :description="$t('pipeline.noEntries')" />
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { tr } from '@/i18n/catalog'
import {
  createCustomPipeline,
  deleteCustomPipeline,
  exportCustomPipeline,
  getCustomPipeline,
  getCustomPipelineSearchSpace,
  getCustomPipelines,
  getPipelineCatalog,
  getPipelineTemplates,
  getReproductionCatalog,
  importCustomPipeline,
  installPipelineTemplate,
  saveCustomPipelineSearchSpace,
  updateCustomPipeline,
  validatePipeline,
} from '@/api'

const { t } = useI18n()

const NODE_W = 178
const NODE_H = 86
const CANVAS_MAX_X = 2600
const CANVAS_MAX_Y = 1800

const KIND_COLORS = {
  preprocess: '#409eff',
  feature: '#67c23a',
  merge: '#e6a23c',
  decoder: '#9254de',
}
const KIND_COLUMN = { preprocess: 0, feature: 1, merge: 2, decoder: 3 }
const KIND_META = computed(() => ({
  preprocess: { label: t('pipeline.kindPreprocess'), color: KIND_COLORS.preprocess },
  feature: { label: t('pipeline.kindFeature'), color: KIND_COLORS.feature },
  merge: { label: t('pipeline.kindMerge'), color: KIND_COLORS.merge },
  decoder: { label: t('pipeline.kindDecoder'), color: KIND_COLORS.decoder },
}))
const IO_LABELS = computed(() => ({
  epochs: t('pipeline.ioEpochs'),
  features: t('pipeline.ioFeatures'),
  labels: t('pipeline.ioLabels'),
  covariance: t('pipeline.ioCovariance'),
  probabilities: t('pipeline.ioProbabilities'),
  any: t('pipeline.ioAny'),
}))
const PALETTE_ORDER = ['preprocess', 'feature', 'merge', 'decoder']
const FUSION_TYPES = new Set(['merge.average', 'merge.vote'])

const TEMPLATES = [
  {
    key: 'csp_lda',
    label: 'pipeline.templateCspLdaLabel',
    name: 'pipeline.templateCspLdaName',
    description: 'pipeline.templateCspLdaDesc',
    nodes: [
      ['preprocess.bandpass', 40, 90],
      ['feature.csp', 320, 90],
      ['decoder.lda', 600, 90],
    ],
    edges: [[0, 1], [1, 2]],
  },
  {
    key: 'dual_branch',
    label: 'pipeline.templateDualBranchLabel',
    name: 'pipeline.templateDualBranchName',
    description: 'pipeline.templateDualBranchDesc',
    nodes: [
      ['preprocess.bandpass', 40, 150],
      ['feature.csp', 320, 50],
      ['feature.channel_std', 320, 230],
      ['merge.concat', 600, 140],
      ['decoder.lda', 880, 140],
    ],
    edges: [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]],
  },
  {
    key: 'fbcsp_lda',
    label: 'pipeline.templateFbcspLabel',
    name: 'pipeline.templateFbcspName',
    description: 'pipeline.templateFbcspDesc',
    nodes: [
      ['preprocess.mean_center', 40, 90],
      ['feature.fbcsp', 320, 90],
      ['decoder.lda', 600, 90],
    ],
    edges: [[0, 1], [1, 2]],
  },
  {
    key: 'riemann',
    label: 'pipeline.templateRiemannLabel',
    name: 'pipeline.templateRiemannName',
    description: 'pipeline.templateRiemannDesc',
    nodes: [
      ['preprocess.bandpass', 40, 90],
      ['decoder.riemann_mdm', 320, 90],
    ],
    edges: [[0, 1]],
  },
  {
    key: 'eegnet',
    label: 'pipeline.templateEegnetLabel',
    name: 'pipeline.templateEegnetName',
    description: 'pipeline.templateEegnetDesc',
    nodes: [
      ['preprocess.bandpass', 40, 90],
      ['decoder.eegnet', 320, 90],
    ],
    edges: [[0, 1]],
  },
  {
    key: 'xdawn_erp',
    label: 'pipeline.templateXdawnLabel',
    name: 'pipeline.templateXdawnName',
    description: 'pipeline.templateXdawnDesc',
    nodes: [
      ['preprocess.bandpass', 40, 90],
      ['preprocess.xdawn', 320, 90],
      ['feature.erp_window', 600, 90],
      ['feature.standardize', 880, 90],
      ['decoder.lda', 1160, 90],
    ],
    edges: [[0, 1], [1, 2], [2, 3], [3, 4]],
  },
  {
    key: 'bandpower_lda',
    label: 'pipeline.templateBandpowerLabel',
    name: 'pipeline.templateBandpowerName',
    description: 'pipeline.templateBandpowerDesc',
    nodes: [
      ['preprocess.car', 40, 90],
      ['feature.bandpower', 320, 90],
      ['feature.standardize', 600, 90],
      ['decoder.lda', 880, 90],
    ],
    edges: [[0, 1], [1, 2], [2, 3]],
  },
  {
    key: 'fusion_dual',
    label: 'pipeline.templateFusionLabel',
    name: 'pipeline.templateFusionName',
    description: 'pipeline.templateFusionDesc',
    nodes: [
      ['preprocess.bandpass', 40, 40, { low: 8, high: 30 }],
      ['feature.csp', 280, 40, { n_components: 4 }],
      ['feature.standardize', 520, 40],
      ['decoder.lda', 760, 40],
      ['preprocess.bandpass', 280, 200, { low: 16, high: 30 }],
      ['feature.channel_std', 520, 200],
      ['decoder.nearest_centroid', 760, 200],
      ['merge.average', 1000, 110],
    ],
    edges: [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [3, 7], [6, 7]],
  },
  {
    key: 'filterbank_bandpower',
    label: 'pipeline.templateFilterbankLabel',
    name: 'pipeline.templateFilterbankName',
    description: 'pipeline.templateFilterbankDesc',
    nodes: [
      ['preprocess.filter_bank_split', 40, 110, { low: 4, high: 32, n_bands: 3 }],
      ['feature.bandpower', 320, 20, { low: 4, high: 13 }],
      ['feature.bandpower', 320, 130, { low: 13, high: 22 }],
      ['feature.bandpower', 320, 240, { low: 22, high: 32 }],
      ['merge.concat', 600, 130],
      ['feature.standardize', 880, 130],
      ['decoder.lda', 1160, 130],
    ],
    edges: [[0, 1, 'band_1'], [0, 2, 'band_2'], [0, 3, 'band_3'], [1, 4], [2, 4], [3, 4], [4, 5], [5, 6]],
  },
]

const catalog = ref([])
const customs = ref([])
const templateKey = ref('')
const loadedId = ref(null)
const validating = ref(false)
const saving = ref(false)
const nodeQuery = ref('')
const viewportEl = ref(null)
const surfaceEl = ref(null)
const pendingSource = ref('')
const pendingPort = ref('')
const selectedNodeId = ref('')
const selectedEdgeIndex = ref(-1)
const validation = ref(null)
const graph = reactive({ id: null, name: t('pipeline.untitled'), description: '', nodes: [], edges: [] })
const drag = reactive({ id: '', offsetX: 0, offsetY: 0 })
const router = useRouter()
const spaceDialog = ref(false)
const spaceLoading = ref(false)
const spaceSaving = ref(false)
const spaceRows = ref([])
const spaceTotal = ref(0)
const savedSignature = ref('')
const marketVisible = ref(false)
const marketTab = ref('templates')
const marketLoading = ref(false)
const marketTemplates = ref([])
const marketInstalling = ref('')
const reproductionLoading = ref(false)
const reproductionEntries = ref([])
const templateFileInput = ref(null)

const specByType = computed(() => Object.fromEntries(catalog.value.map((item) => [item.type, item])))
const selectedNode = computed(() => graph.nodes.find((node) => node.id === selectedNodeId.value) || null)
const selectedSpec = computed(() => (selectedNode.value ? specByType.value[selectedNode.value.type] : null))
const selectedEdge = computed(() => (
  selectedEdgeIndex.value >= 0 ? graph.edges[selectedEdgeIndex.value] || null : null
))
const decoderNodes = computed(() => graph.nodes.filter((node) => isDecoder(node)))
const fusionNode = computed(() => graph.nodes.find((node) => isFusion(node)) || null)
const decoderLabel = computed(() => {
  const names = decoderNodes.value.map((node) => tr(node.label))
  if (!names.length) return ''
  return fusionNode.value ? `${names.join(' + ')} → ${tr(fusionNode.value.label)}` : names.join(' + ')
})
const validationDecoderLabel = computed(() => {
  const ids = validation.value?.decoders
    || (validation.value?.decoder ? [validation.value.decoder] : [])
  return ids.map((id) => nodeLabel(id)).join(' + ') || '-'
})
const matchedNodes = computed(() => {
  const query = nodeQuery.value.trim().toLowerCase()
  if (!query) return catalog.value
  return catalog.value.filter((item) => (
    `${item.label} ${tr(item.label)} ${item.type} ${item.description || ''} ${tr(item.description || '')} ${item.paradigm || ''}`
      .toLowerCase()
      .includes(query)
  ))
})
const filteredCount = computed(() => matchedNodes.value.length)
const paletteGroups = computed(() => PALETTE_ORDER.map((kind) => ({
  kind,
  label: KIND_META.value[kind].label,
  color: KIND_META.value[kind].color,
  items: matchedNodes.value.filter((item) => item.kind === kind),
})))
const paramFields = computed(() => {
  const spec = selectedSpec.value
  if (!spec) return []
  const properties = spec.params_schema?.properties || {}
  const labels = Object.fromEntries((spec.params || []).map((param) => [param.name, param.label]))
  return Object.entries(properties).map(([name, property]) => ({
    name,
    label: labels[name] || property.title || name,
    type: property.type || 'string',
    schema: property,
  }))
})
const spaceEnabledCount = computed(() => spaceRows.value.filter((row) => row.enabled).length)
const canvasSize = computed(() => ({
  width: Math.max(900, ...graph.nodes.map((node) => node.x + NODE_W + 90)),
  height: Math.max(560, ...graph.nodes.map((node) => node.y + NODE_H + 90)),
}))
const topologyText = computed(() => {
  const order = validation.value?.topology || []
  if (!order.length) return '-'
  return order.map((id) => nodeLabel(id)).join(' → ')
})

function kindOf(node) {
  return specByType.value[node.type]?.kind || 'feature'
}
function colorFor(node) {
  return KIND_META.value[kindOf(node)]?.color || '#909399'
}
function kindLabel(node) {
  return KIND_META.value[kindOf(node)]?.label || kindOf(node)
}
function isDecoder(node) {
  return kindOf(node) === 'decoder'
}
function isFusion(node) {
  return FUSION_TYPES.has(node?.type)
}
function isMultiInput(node) {
  return node?.type === 'merge.concat' || isFusion(node)
}
function canOutput(node) {
  return Boolean(node) && !isFusion(node)
}
function nodeLabel(id) {
  const label = graph.nodes.find((node) => node.id === id)?.label
  return label ? tr(label) : id
}
function nodeStyle(node) {
  return { left: `${node.x}px`, top: `${node.y}px`, '--node-color': colorFor(node) }
}
function formatHz(value) {
  return Number(value.toFixed(1)).toString()
}
function ioLabel(type) {
  return IO_LABELS.value[type] || type
}
function portsFor(node) {
  if (!node) return []
  if (node.type === 'preprocess.filter_bank_split') {
    const low = Number(node.params?.low)
    const high = Number(node.params?.high)
    const nBands = Math.round(Number(node.params?.n_bands))
    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return []
    if (!Number.isFinite(nBands) || nBands < 2 || nBands > 8) return []
    const edges = Array.from(
      { length: nBands + 1 },
      (_, index) => low + ((high - low) * index) / nBands,
    )
    return Array.from({ length: nBands }, (_, index) => ({
      name: `band_${index + 1}`,
      label: `${formatHz(edges[index])}–${formatHz(edges[index + 1])} Hz`,
      type: 'epochs',
    }))
  }
  return specByType.value[node.type]?.output_ports || []
}
function portTop(node, portName) {
  const ports = portsFor(node)
  if (!ports.length) return NODE_H / 2
  const index = ports.findIndex((port) => port.name === portName)
  if (index < 0) return NODE_H / 2
  const headerHeight = 34
  return headerHeight + ((NODE_H - headerHeight) * (index + 0.5)) / ports.length
}
function edgePorts(edge) {
  if (!edge) return []
  const source = graph.nodes.find((node) => node.id === edge.source)
  return portsFor(source)
}
function edgePortLabel(edge) {
  const port = edgePorts(edge).find((item) => item.name === edge.source_port)
  return port ? port.label : edge.source_port
}
function edgeKey(edge) {
  return `${edge.source}-${edge.source_port || '_'}-${edge.target}`
}
function precisionFor(schema) {
  const step = Number(schema?.step ?? 0.1)
  if (!Number.isFinite(step) || step >= 1) return 0
  if (step >= 0.1) return 1
  if (step >= 0.01) return 2
  if (step >= 0.001) return 3
  return 6
}
function formatParamValue(value) {
  if (typeof value === 'boolean') return value ? t('pipeline.on') : t('pipeline.off')
  return String(value)
}
function summary(node) {
  const params = node.params || {}
  switch (node.type) {
    case 'preprocess.bandpass':
      return `${params.low}–${params.high} Hz · order ${params.order}`
    case 'preprocess.filter_bank_split':
      return t('pipeline.summarySubBands', { low: params.low, high: params.high, count: params.n_bands })
    case 'feature.bandpower':
      return `${params.low}–${params.high} Hz${params.relative ? ` · ${t('pipeline.summaryRelative')}` : ''}`
    case 'feature.csp':
      return t('pipeline.summaryCsp', { count: params.n_components })
    case 'merge.average':
      return t('pipeline.summaryAverage')
    case 'merge.vote':
      return t('pipeline.summaryVote')
    case 'feature.fbcsp':
      return t('pipeline.summaryFbcsp', { count: params.n_components })
    case 'decoder.lda':
      return `shrinkage=${params.shrinkage}`
    case 'decoder.riemann_mdm':
      return `${params.metric} · ${params.covariance}`
    case 'decoder.tangent_space_lr':
      return `${params.metric} · C=${params.C}`
    case 'decoder.eegnet':
    case 'decoder.shallowconvnet':
      return `${params.max_epochs} epochs · lr ${params.learning_rate}`
    default:
      return genericSummary(node)
  }
}
function genericSummary(node) {
  const spec = specByType.value[node.type]
  const parts = []
  for (const param of spec?.params || []) {
    const value = node.params?.[param.name]
    if (value === undefined || value === null || value === '') continue
    parts.push(`${tr(param.label)}=${formatParamValue(value)}`)
    if (parts.length >= 2) break
  }
  return parts.join(' · ')
}

function defaultParams(type) {
  const spec = specByType.value[type]
  const defaults = Object.fromEntries(
    (spec?.params || []).map((param) => [param.name, param.default]),
  )
  const properties = spec?.params_schema?.properties || {}
  for (const [name, property] of Object.entries(properties)) {
    if (defaults[name] === undefined && 'default' in property) defaults[name] = property.default
  }
  return defaults
}

function resetParams() {
  if (!selectedNode.value) return
  selectedNode.value.params = defaultParams(selectedNode.value.type)
  validation.value = null
}

function makeId(type, taken = graph.nodes) {
  const base = type.split('.').pop().replace(/[^A-Za-z0-9_-]/g, '') || 'node'
  const ids = new Set(taken.map((node) => node.id))
  let index = 1
  while (ids.has(`${base}_${index}`)) index += 1
  return `${base}_${index}`
}

function makeNode(type, x, y, taken, params = {}) {
  const spec = specByType.value[type]
  if (!spec) throw new Error(`unknown node type: ${type}`)
  return {
    id: makeId(type, taken),
    type,
    label: spec.label,
    params: { ...defaultParams(type), ...params },
    x,
    y,
  }
}

function addNode(spec) {
  if (graph.nodes.length >= 32) {
    ElMessage.warning(t('pipeline.maxNodes'))
    return
  }
  if (isFusion(spec) && graph.nodes.some((node) => isFusion(node))) {
    ElMessage.warning(t('pipeline.singleFusion'))
    return
  }
  const sameKind = graph.nodes.filter((node) => kindOf(node) === spec.kind).length
  const x = 40 + (KIND_COLUMN[spec.kind] ?? 0) * 250
  const y = 50 + (sameKind % 7) * 104
  const node = makeNode(spec.type, x, y, graph.nodes)
  graph.nodes.push(node)
  selectNode(node.id)
  validation.value = null
  nextTick(() => {
    viewportEl.value?.scrollTo({ left: Math.max(0, x - 160), top: Math.max(0, y - 60), behavior: 'smooth' })
  })
}

function applyTemplate(key) {
  const template = TEMPLATES.find((item) => item.key === key)
  if (!template) return
  templateKey.value = ''
  if (!catalog.value.length) return
  const nodes = []
  for (const [type, x, y, params] of template.nodes) {
    nodes.push(makeNode(type, x, y, nodes, params || {}))
  }
  graph.id = null
  graph.name = t(template.name)
  graph.description = t(template.description)
  graph.nodes = nodes
  graph.edges = template.edges.map(([source, target, sourcePort]) => {
    const edge = { source: nodes[source].id, target: nodes[target].id }
    if (sourcePort) edge.source_port = sourcePort
    return edge
  })
  loadedId.value = null
  savedSignature.value = ''
  clearSelection()
  validation.value = null
  nextTick(() => viewportEl.value?.scrollTo({ left: 0, top: 0 }))
}

function newGraph() {
  graph.id = null
  graph.name = t('pipeline.untitled')
  graph.description = ''
  graph.nodes = []
  graph.edges = []
  loadedId.value = null
  savedSignature.value = ''
  spaceTotal.value = 0
  pendingSource.value = ''
  pendingPort.value = ''
  clearSelection()
  validation.value = null
}

function clearSelection() {
  selectedNodeId.value = ''
  selectedEdgeIndex.value = -1
}

function selectNode(id) {
  selectedNodeId.value = id
  selectedEdgeIndex.value = -1
}

function selectEdge(index) {
  selectedEdgeIndex.value = index
  selectedNodeId.value = ''
}

function beginConnect(id, port = '') {
  const node = graph.nodes.find((item) => item.id === id)
  if (!node || !canOutput(node)) return
  if (pendingSource.value === id && pendingPort.value === port) {
    pendingSource.value = ''
    pendingPort.value = ''
  } else {
    pendingSource.value = id
    pendingPort.value = port || ''
  }
  selectNode(id)
}

function wouldCycle(source, target) {
  const adjacency = new Map()
  for (const node of graph.nodes) adjacency.set(node.id, [])
  for (const edge of graph.edges) adjacency.get(edge.source)?.push(edge.target)
  const stack = [target]
  const seen = new Set()
  while (stack.length) {
    const current = stack.pop()
    if (current === source) return true
    if (seen.has(current)) continue
    seen.add(current)
    for (const next of adjacency.get(current) || []) stack.push(next)
  }
  return false
}

function connectTo(targetId) {
  const source = pendingSource.value
  if (!source) {
    ElMessage.info(t('pipeline.selectSourceFirst'))
    return
  }
  if (source === targetId) return
  const sourceNode = graph.nodes.find((node) => node.id === source)
  const targetNode = graph.nodes.find((node) => node.id === targetId)
  if (!sourceNode || !targetNode) return
  if (!canOutput(sourceNode)) {
    ElMessage.warning(t('pipeline.fusionIsSink'))
    return
  }
  if (isFusion(targetNode)) {
    if (!isDecoder(sourceNode)) {
      ElMessage.warning(t('pipeline.fusionOnlyDecoders'))
      return
    }
    if (graph.nodes.some((node) => isFusion(node) && node.id !== targetId)) {
      ElMessage.warning(t('pipeline.singleFusion'))
      return
    }
  } else if (isDecoder(sourceNode)) {
    ElMessage.warning(t('pipeline.decoderOnlyFusion'))
    return
  }
  const sourcePorts = portsFor(sourceNode)
  const sourcePort = sourcePorts.length ? pendingPort.value : ''
  if (sourcePorts.length && !sourcePorts.some((port) => port.name === sourcePort)) {
    ElMessage.warning(t('pipeline.selectPortFirst'))
    return
  }
  const sourceSpec = specByType.value[sourceNode.type]
  const targetSpec = specByType.value[targetNode.type]
  const outputType = sourcePorts[0]?.type || sourceSpec?.outputs?.[0] || ''
  const targetInputs = targetSpec?.inputs || []
  if (outputType && targetInputs.length
    && !targetInputs.includes(outputType) && !targetInputs.includes('any')) {
    ElMessage.warning(t('pipeline.inputTypeMismatch', {
      source: sourceNode.label,
      output: ioLabel(outputType),
      target: targetNode.label,
      accepted: targetInputs.map(ioLabel).join(' / '),
    }))
    return
  }
  if (graph.edges.some((edge) => (
    edge.source === source
    && edge.target === targetId
    && (edge.source_port || '') === sourcePort
  ))) {
    ElMessage.warning(t('pipeline.edgeExists'))
    return
  }
  if (wouldCycle(source, targetId)) {
    ElMessage.warning(t('pipeline.edgeCycle'))
    return
  }
  if (!isMultiInput(targetNode)
    && graph.edges.some((edge) => edge.target === targetId)) {
    ElMessage.warning(t('pipeline.singleInputOnly'))
    return
  }
  const edge = { source, target: targetId }
  if (sourcePort) edge.source_port = sourcePort
  graph.edges.push(edge)
  pendingSource.value = ''
  pendingPort.value = ''
  validation.value = null
  selectNode(targetId)
}

function startDrag(event, node) {
  if (event.button !== 0) return
  selectNode(node.id)
  const rect = surfaceEl.value?.getBoundingClientRect()
  if (!rect) return
  drag.id = node.id
  drag.offsetX = event.clientX - rect.left - node.x
  drag.offsetY = event.clientY - rect.top - node.y
  window.addEventListener('pointermove', onDragMove)
  window.addEventListener('pointerup', stopDrag, { once: true })
}

function onDragMove(event) {
  if (!drag.id || !surfaceEl.value) return
  const node = graph.nodes.find((item) => item.id === drag.id)
  if (!node) return
  const rect = surfaceEl.value.getBoundingClientRect()
  const x = event.clientX - rect.left - drag.offsetX
  const y = event.clientY - rect.top - drag.offsetY
  node.x = Math.max(0, Math.min(CANVAS_MAX_X, Math.round(x / 8) * 8))
  node.y = Math.max(0, Math.min(CANVAS_MAX_Y, Math.round(y / 8) * 8))
}

function stopDrag() {
  drag.id = ''
  window.removeEventListener('pointermove', onDragMove)
}

function removeNode(id) {
  graph.nodes = graph.nodes.filter((node) => node.id !== id)
  graph.edges = graph.edges.filter((edge) => edge.source !== id && edge.target !== id)
  if (pendingSource.value === id) {
    pendingSource.value = ''
    pendingPort.value = ''
  }
  if (selectedNodeId.value === id) selectedNodeId.value = ''
  selectedEdgeIndex.value = -1
  validation.value = null
}

function removeSelectedNode() {
  if (!selectedNodeId.value) return
  removeNode(selectedNodeId.value)
}

function removeSelectedEdge() {
  if (selectedEdgeIndex.value < 0) return
  graph.edges.splice(selectedEdgeIndex.value, 1)
  selectedEdgeIndex.value = -1
  validation.value = null
}

function deleteSelection() {
  if (selectedNodeId.value) removeSelectedNode()
  else if (selectedEdgeIndex.value >= 0) removeSelectedEdge()
}

function onKeydown(event) {
  const tag = event.target?.tagName
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || event.target?.isContentEditable) return
  if (event.key === 'Delete' || event.key === 'Backspace') {
    if (!selectedNodeId.value && selectedEdgeIndex.value < 0) return
    event.preventDefault()
    deleteSelection()
  } else if (event.key === 'Escape') {
    pendingSource.value = ''
    pendingPort.value = ''
    clearSelection()
  }
}

function edgePoints(edge) {
  const source = graph.nodes.find((node) => node.id === edge.source)
  const target = graph.nodes.find((node) => node.id === edge.target)
  if (!source || !target) return null
  const x1 = source.x + NODE_W
  const y1 = source.y + portTop(source, edge.source_port)
  const x2 = target.x
  const y2 = target.y + NODE_H / 2
  const bend = Math.max(60, Math.abs(x2 - x1) / 2)
  return { x1, y1, x2, y2, bend }
}
function edgePath(edge) {
  const points = edgePoints(edge)
  if (!points) return ''
  const { x1, y1, x2, y2, bend } = points
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`
}
function edgeMid(edge) {
  const points = edgePoints(edge)
  if (!points) return { x: 0, y: 0 }
  return { x: (points.x1 + points.x2) / 2, y: (points.y1 + points.y2) / 2 }
}

function autoLayout() {
  if (!graph.nodes.length) return
  const indegree = new Map(graph.nodes.map((node) => [node.id, 0]))
  const outgoing = new Map(graph.nodes.map((node) => [node.id, []]))
  for (const edge of graph.edges) {
    indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1)
    outgoing.get(edge.source)?.push(edge.target)
  }
  const level = new Map(graph.nodes.map((node) => [node.id, 0]))
  const queue = graph.nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id)
  const order = []
  while (queue.length) {
    const current = queue.shift()
    order.push(current)
    for (const target of outgoing.get(current) || []) {
      level.set(target, Math.max(level.get(target) || 0, (level.get(current) || 0) + 1))
      indegree.set(target, indegree.get(target) - 1)
      if (indegree.get(target) === 0) queue.push(target)
    }
  }
  if (order.length !== graph.nodes.length) {
    ElMessage.warning(t('pipeline.layoutCycle'))
    return
  }
  const groups = new Map()
  for (const node of graph.nodes) {
    const depth = level.get(node.id) || 0
    if (!groups.has(depth)) groups.set(depth, [])
    groups.get(depth).push(node)
  }
  for (const [depth, nodes] of [...groups.entries()].sort((left, right) => left[0] - right[0])) {
    nodes.forEach((node, index) => {
      node.x = 40 + depth * 240
      node.y = 40 + index * 104
    })
  }
  ElMessage.success(t('pipeline.layoutDone'))
}

function buildSpec() {
  return {
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      label: node.label,
      params: node.params,
      x: Math.round(node.x),
      y: Math.round(node.y),
    })),
    edges: graph.edges.map((edge) => {
      const item = { source: edge.source, target: edge.target }
      if (edge.source_port) item.source_port = edge.source_port
      return item
    }),
  }
}

async function validateGraph() {
  validating.value = true
  try {
    const result = await validatePipeline({
      name: graph.name.trim() || 'draft',
      description: graph.description,
      spec: buildSpec(),
    })
    validation.value = result
    if (result.valid) ElMessage.success(t('pipeline.validateSuccess', { nodes: result.nodes, edges: result.edges }))
    else ElMessage.warning(result.error)
  } finally {
    validating.value = false
  }
}

function applyItem(item) {
  graph.id = item.id
  graph.name = item.name
  graph.description = item.description || ''
  graph.nodes = (item.spec.nodes || []).map((node, index) => ({
    ...node,
    params: { ...defaultParams(node.type), ...(node.params || {}) },
    x: node.x ?? 40 + (index % 4) * 240,
    y: node.y ?? 40 + Math.floor(index / 4) * 104,
  }))
  graph.edges = (item.spec.edges || []).map((edge) => ({ ...edge }))
  loadedId.value = item.id
  pendingSource.value = ''
  clearSelection()
  savedSignature.value = graphSignature()
  validation.value = {
    valid: true,
    nodes: graph.nodes.length,
    edges: graph.edges.length,
    topology: item.spec.topology || [],
    decoder: item.spec.decoder,
    decoders: item.spec.decoders,
    prediction: item.spec.prediction,
  }
}

async function refreshCustoms() {
  const response = await getCustomPipelines()
  customs.value = response.items
}

function graphSignature() {
  return JSON.stringify({
    name: graph.name,
    description: graph.description,
    spec: buildSpec(),
  })
}

const SPACE_KIND_LABELS = computed(() => ({
  int: t('pipeline.spaceKindInt'),
  float: t('pipeline.spaceKindFloat'),
  select: t('pipeline.spaceKindSelect'),
}))

function spaceKindLabel(kind) {
  return SPACE_KIND_LABELS.value[kind] || kind
}

function buildSpaceRows(dims = []) {
  const saved = new Map(dims.map((dim) => [`${dim.node_id}::${dim.param}`, dim]))
  const rows = []
  for (const node of graph.nodes) {
    const spec = specByType.value[node.type]
    if (!spec || spec.plugin) continue
    for (const param of spec.params || []) {
      if (!['int', 'float', 'select'].includes(param.type)) continue
      if (param.name === 'n_bands' && (spec.dynamic_ports || node.type === 'preprocess.filter_bank_split')) continue
      const options = (param.options || []).map(String)
      if (param.type === 'select' && options.length < 2) continue
      const savedDim = saved.get(`${node.id}::${param.name}`)
      const current = node.params?.[param.name] ?? param.default
      rows.push({
        key: `${node.id}::${param.name}`,
        node_id: node.id,
        node_label: node.label,
        node_type: node.type,
        param: param.name,
        label: param.label || param.name,
        kind: param.type,
        current,
        schemaMin: param.min,
        schemaMax: param.max,
        enabled: Boolean(savedDim),
        low: savedDim?.low ?? param.min,
        high: savedDim?.high ?? param.max,
        step: savedDim && 'step' in savedDim ? savedDim.step : (param.step ?? (param.type === 'int' ? 1 : undefined)),
        options,
        selectedOptions: savedDim?.options ? [...savedDim.options] : [...options],
      })
    }
  }
  return rows
}

async function openSpace() {
  if (!graph.id) {
    ElMessage.warning(t('pipeline.saveFirst'))
    return
  }
  if (graphSignature() !== savedSignature.value) {
    try {
      await ElMessageBox.confirm(
        t('pipeline.unsavedChanges'),
        t('pipeline.saveChangesTitle'),
        { type: 'warning', confirmButtonText: t('pipeline.saveFirstButton'), cancelButtonText: t('common.cancel') },
      )
    } catch {
      return
    }
    const item = await saveGraph()
    if (!item) return
  }
  spaceLoading.value = true
  spaceDialog.value = true
  try {
    const response = await getCustomPipelineSearchSpace(graph.id)
    spaceRows.value = buildSpaceRows(response.dims || [])
    spaceTotal.value = response.total || 0
  } catch {
    spaceDialog.value = false
  } finally {
    spaceLoading.value = false
  }
}

function collectSpaceDims() {
  const enabled = spaceRows.value.filter((row) => row.enabled)
  if (!enabled.length) throw new Error(t('pipeline.spaceNeedOne'))
  if (enabled.length > 16) throw new Error(t('pipeline.spaceMaxDims'))
  const dims = []
  for (const row of enabled) {
    const name = `${tr(row.node_label)} · ${tr(row.label)}`
    if (row.kind === 'select') {
      if (row.selectedOptions.length < 2) throw new Error(t('pipeline.spaceMinOptions', { name }))
      if (!row.selectedOptions.includes(String(row.current))) {
        throw new Error(t('pipeline.spaceKeepCurrent', { name, current: row.current }))
      }
      dims.push({ node_id: row.node_id, param: row.param, options: row.selectedOptions })
      continue
    }
    if (row.low === undefined || row.low === null || row.high === undefined || row.high === null) {
      throw new Error(t('pipeline.spaceNeedBounds', { name }))
    }
    if (row.low >= row.high) throw new Error(t('pipeline.spaceLowHigh', { name }))
    if (Number(row.current) < row.low || Number(row.current) > row.high) {
      throw new Error(t('pipeline.spaceOutOfRange', { name, current: row.current }))
    }
    if (row.kind === 'int') {
      if (!Number.isInteger(row.low) || !Number.isInteger(row.high)) {
        throw new Error(t('pipeline.spaceIntBounds', { name }))
      }
      if (!Number.isInteger(row.step) || row.step <= 0) {
        throw new Error(t('pipeline.spaceIntStep', { name }))
      }
    } else if (row.step !== undefined && row.step !== null && row.step !== '' && row.step <= 0) {
      throw new Error(t('pipeline.spaceFloatStep', { name }))
    }
    const dim = { node_id: row.node_id, param: row.param, low: row.low, high: row.high }
    if (row.step !== undefined && row.step !== null && row.step !== '') dim.step = row.step
    dims.push(dim)
  }
  return dims
}

async function saveSpace(navigate = false) {
  if (!graph.id) return
  let dims
  try {
    dims = collectSpaceDims()
  } catch (error) {
    ElMessage.warning(error.message)
    return
  }
  spaceSaving.value = true
  try {
    const response = await saveCustomPipelineSearchSpace(graph.id, dims)
    spaceTotal.value = response.total || 0
    spaceRows.value = buildSpaceRows(response.dims || [])
    spaceDialog.value = false
    ElMessage.success(t('pipeline.spaceSaved', { count: response.total }))
    if (navigate) router.push({ path: '/hpo', query: { template: `custom:${graph.id}` } })
  } catch {
    // API interceptor already surfaced the reason
  } finally {
    spaceSaving.value = false
  }
}

async function clearSpace() {
  if (!graph.id) return
  try {
    await ElMessageBox.confirm(
      t('pipeline.spaceClearConfirm'),
      t('pipeline.spaceClearTitle'),
      { type: 'warning', confirmButtonText: t('pipeline.clear'), cancelButtonText: t('common.cancel') },
    )
  } catch {
    return
  }
  spaceSaving.value = true
  try {
    await saveCustomPipelineSearchSpace(graph.id, [])
    spaceRows.value = buildSpaceRows([])
    spaceTotal.value = 0
    spaceDialog.value = false
    ElMessage.success(t('pipeline.spaceCleared'))
  } catch {
    // API interceptor already surfaced the reason
  } finally {
    spaceSaving.value = false
  }
}

async function saveGraph() {
  if (!graph.name.trim()) {
    ElMessage.warning(t('pipeline.nameRequired'))
    return null
  }
  saving.value = true
  try {
    const payload = {
      name: graph.name.trim(),
      description: graph.description.trim(),
      spec: buildSpec(),
    }
    const item = graph.id
      ? await updateCustomPipeline(graph.id, payload)
      : await createCustomPipeline(payload)
    applyItem(item)
    await refreshCustoms()
    ElMessage.success(t('pipeline.savedMessage', { id: item.id }))
    return item
  } catch {
    // API interceptor already surfaced the reason
    return null
  } finally {
    saving.value = false
  }
}

async function loadCustom(id) {
  if (!id) return
  try {
    const item = await getCustomPipeline(id)
    applyItem(item)
    nextTick(() => viewportEl.value?.scrollTo({ left: 0, top: 0 }))
    ElMessage.success(t('pipeline.loadedMessage', { name: item.name }))
  } catch {
    loadedId.value = null
    return
  }
  try {
    const space = await getCustomPipelineSearchSpace(id)
    spaceTotal.value = space.total || 0
  } catch {
    spaceTotal.value = 0
  }
}

async function removeCurrent() {
  if (!graph.id) return
  try {
    await ElMessageBox.confirm(t('pipeline.deleteConfirm', { name: graph.name }), t('pipeline.deleteTitle'), {
      type: 'warning', confirmButtonText: t('common.delete'), cancelButtonText: t('common.cancel'),
    })
  } catch {
    return
  }
  try {
    await deleteCustomPipeline(graph.id)
    await refreshCustoms()
    newGraph()
    ElMessage.success(t('pipeline.deleted'))
  } catch {
    // API interceptor already surfaced the reason
  }
}

async function openMarket() {
  marketVisible.value = true
  if (marketTemplates.value.length && reproductionEntries.value.length) return
  marketLoading.value = true
  reproductionLoading.value = true
  try {
    const [templates, reproduction] = await Promise.all([
      marketTemplates.value.length ? null : getPipelineTemplates(),
      reproductionEntries.value.length ? null : getReproductionCatalog(),
    ])
    if (templates) marketTemplates.value = templates.items
    if (reproduction) reproductionEntries.value = reproduction.items
  } finally {
    marketLoading.value = false
    reproductionLoading.value = false
  }
}

function loadReproductionToCanvas(entry) {
  const template = marketTemplates.value.find((item) => item.id === entry.template_id || item.id === entry.id)
  if (template) loadTemplateToCanvas(template)
}

function loadTemplateToCanvas(item) {
  if (!catalog.value.length) return
  applyItem({ id: null, name: item.name, description: item.description, spec: item.spec })
  savedSignature.value = ''
  marketVisible.value = false
  nextTick(() => viewportEl.value?.scrollTo({ left: 0, top: 0 }))
  ElMessage.success(t('templates.loaded', { name: item.name }))
}

async function installTemplate(item) {
  marketInstalling.value = item.id
  try {
    const result = await installPipelineTemplate(item.id, { name: item.name })
    applyItem(result.item)
    await refreshCustoms()
    marketVisible.value = false
    ElMessage.success(t('templates.installed', { name: result.item.name }))
  } catch {
    // API interceptor already surfaced the reason
  } finally {
    marketInstalling.value = ''
  }
}

async function exportCurrent() {
  if (!graph.id) return
  try {
    const payload = await exportCustomPipeline(graph.id)
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${(payload.name || 'pipeline').replace(/[\\/:*?"<>|]+/g, '_')}.neurokiln.json`
    link.click()
    URL.revokeObjectURL(url)
  } catch {
    // API interceptor already surfaced the reason
  }
}

function triggerTemplateImport() {
  templateFileInput.value?.click()
}

async function onTemplateFile(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  try {
    const payload = JSON.parse(await file.text())
    const result = await importCustomPipeline(payload)
    applyItem(result.item)
    await refreshCustoms()
    ElMessage.success(t('templates.imported', { name: result.item.name }))
  } catch (error) {
    if (error instanceof SyntaxError) ElMessage.error(t('pipeline.invalidTemplateJson'))
  }
}

watch(
  () => graph.nodes.map((node) => [node.id, node.type, JSON.stringify(node.params)]),
  () => {
    if (!graph.edges.some((edge) => edge.source_port)) return
    const kept = graph.edges.filter((edge) => {
      if (!edge.source_port) return true
      const source = graph.nodes.find((node) => node.id === edge.source)
      return Boolean(source) && portsFor(source).some((port) => port.name === edge.source_port)
    })
    if (kept.length !== graph.edges.length) {
      graph.edges = kept
      validation.value = null
    }
  },
)

onMounted(async () => {
  const [catalogResponse, customResponse] = await Promise.all([
    getPipelineCatalog(),
    getCustomPipelines(),
  ])
  catalog.value = catalogResponse.items
  customs.value = customResponse.items
  applyTemplate('csp_lda')
  window.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  stopDrag()
})
</script>

<style scoped>
.page-head { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; justify-content: space-between; }
.head-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.toolbar-label { color: #606266; font-size: 13px; }
.palette { display: flex; flex-direction: column; gap: 8px; }
.palette-head { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 10px; }
.palette-heading { font-size: 13px; font-weight: 600; color: #303133; }
.palette-group { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.palette-kind { width: 56px; font-size: 12px; font-weight: 600; }
.palette-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.palette-chip { margin: 0; }
.palette-plugin {
  margin-left: 4px; padding: 0 3px; font-size: 10px; line-height: 14px;
  color: #f56c6c; border: 1px solid #fbc4c4; border-radius: 3px;
}
.node-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.editor-col { margin-bottom: 12px; }
.canvas-card { overflow: hidden; }
.canvas-head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.hint { font-size: 12px; }
.canvas-viewport {
  height: 640px; overflow: auto; background:
    linear-gradient(rgba(64, 158, 255, .06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(64, 158, 255, .06) 1px, transparent 1px),
    #fbfcfe;
  background-size: 24px 24px;
}
.canvas-surface { position: relative; min-width: 100%; }
.edges { position: absolute; inset: 0; pointer-events: none; overflow: visible; z-index: 1; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 14; pointer-events: stroke; cursor: pointer; }
.edge-line { fill: none; stroke: #a8b3c4; stroke-width: 2; pointer-events: none; }
.edge-line.selected { stroke: #409eff; stroke-width: 3; }
.edge-label {
  font-size: 10px; fill: #a8abb2; pointer-events: none;
  paint-order: stroke; stroke: #fbfcfe; stroke-width: 3px;
}
.node {
  position: absolute; width: 178px; height: 86px; z-index: 2;
  background: #fff; border-radius: 8px; border: 1px solid #dfe5ee;
  border-top: 3px solid var(--node-color);
  box-shadow: 0 3px 10px rgba(31, 45, 61, .08);
  user-select: none; transition: box-shadow .15s ease, border-color .15s ease;
}
.node:hover { box-shadow: 0 6px 16px rgba(31, 45, 61, .14); }
.node.selected { border-color: #409eff; box-shadow: 0 0 0 2px rgba(64, 158, 255, .22); }
.node.pending { box-shadow: 0 0 0 2px rgba(230, 162, 60, .38); }
.node-head {
  height: 34px; display: flex; align-items: center; justify-content: space-between;
  gap: 6px; padding: 0 10px; cursor: grab; border-bottom: 1px solid #f0f2f5;
}
.node-head:active { cursor: grabbing; }
.node-title { font-size: 13px; font-weight: 600; color: #303133; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.node-kind { flex: none; font-size: 10px; line-height: 16px; border-radius: 3px; padding: 0 5px; }
.node-body { padding: 6px 10px 0; }
.node-type { font-size: 10px; color: #a8abb2; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.node-summary { font-size: 11px; color: #606266; margin-top: 3px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.port {
  position: absolute; top: 50%; width: 14px; height: 14px; padding: 0;
  border-radius: 50%; border: 2px solid #fff; background: #909399;
  transform: translateY(-50%); cursor: crosshair; z-index: 3;
  box-shadow: 0 0 0 1px #c0c4cc;
}
.port-in { left: -7px; }
.port-out { right: -7px; background: #409eff; }
.port-out.active { box-shadow: 0 0 0 4px rgba(64, 158, 255, .25); }
.port-anchor {
  position: absolute; top: 50%; right: -7px; transform: translateY(-50%);
  z-index: 3; line-height: 0;
}
.port-anchor .port { position: static; transform: none; }
.port-multi {
  background: #e6a23c;
  box-shadow: 0 0 0 1px #d48806, 0 0 0 5px rgba(230, 162, 60, .16);
}
.port-multi.active { box-shadow: 0 0 0 1px #d48806, 0 0 0 6px rgba(230, 162, 60, .32); }
.empty-canvas {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  color: #a8abb2; font-size: 14px; pointer-events: none;
}
.inspector { min-height: 640px; }
.node-type-text { font-size: 12px; color: #606266; }
.node-desc { margin-bottom: 4px; }
.inspector-actions { margin-top: 16px; }
.validation-head { display: flex; align-items: center; justify-content: space-between; }
.validation-body { margin: 8px 0 0; font-size: 12px; line-height: 1.7; color: #606266; }
.space-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }
.space-node { font-size: 12px; color: #303133; }
.space-type { font-size: 10px; color: #a8abb2; }
.space-range { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.space-num { width: 88px; }
.space-dash { color: #909399; font-size: 12px; }
.space-step-label { margin-left: 4px; color: #909399; font-size: 12px; }
.space-footer { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between; }
.market-desc { margin: 0 0 12px; font-size: 13px; }
.market-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; min-height: 120px; }
.market-card :deep(.el-card__body) { padding: 14px; }
.market-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.market-text { margin: 8px 0; color: #5e6d82; font-size: 12px; line-height: 1.6; }
.market-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.market-paper { margin-top: 6px; color: #909399; font-size: 11px; line-height: 1.6; }
.market-label { display: inline-block; margin-right: 4px; color: #606266; font-weight: 600; }
.market-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 12px; }
@media (max-width: 1199px) {
  .canvas-viewport { height: 500px; }
}
</style>
