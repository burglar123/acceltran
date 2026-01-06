# Scheduler for compute operations with dependency tracking

from collections import deque

from ops import MatrixMultOp, MatrixMultTiledOp, Conv1DOp, Conv1DTiledOp, NonLinearityOp, NonLinearityTiledOp, SoftmaxOp, SoftmaxTiledOp


class Scheduler(object):
	def __init__(self, compute_ops):
		self.compute_ops = compute_ops
		self.dependencies = {}
		self.dependents = {}
		self.ready_queue = deque()
		self.pending = set()
		self.inflight = set()
		self.done = set()

		self._build_graph()

	def _add_dependency(self, op, dep):
		self.dependencies.setdefault(op, set()).add(dep)
		self.dependents.setdefault(dep, set()).add(op)

	def _register_op(self, op):
		if op not in self.dependencies:
			self.dependencies[op] = set()
		if op not in self.dependents:
			self.dependents[op] = set()
		self.pending.add(op)

	def _build_graph(self):
		streams = []
		for item in self.compute_ops:
			if isinstance(item, list):
				for head_ops in item:
					streams.append(head_ops)
			else:
				streams.append([item])

		for stream in streams:
			prev = None
			for op in stream:
				if op is None:
					continue
				self._register_op(op)
				if op.dependencies:
					for dep in op.dependencies:
						self._add_dependency(op, dep)
				elif prev is not None:
					self._add_dependency(op, prev)
				prev = op

		for op in list(self.pending):
			if not self.dependencies.get(op):
				self.ready_queue.append(op)

	def _deps_done(self, op):
		return all(dep in self.done for dep in self.dependencies.get(op, set()))

	def update_done(self):
		new_done = []
		for op in list(self.inflight):
			if op.done:
				self.inflight.remove(op)
				self.done.add(op)
				new_done.append(op)
		for op in new_done:
			for dep in self.dependents.get(op, set()):
				if dep in self.pending and self._deps_done(dep):
					if dep not in self.ready_queue and dep not in self.inflight:
						self.ready_queue.append(dep)

	def step(self, accelerator, buffer_check=None):
		assigned_ops = []
		for _ in range(len(self.ready_queue)):
			op = self.ready_queue[0]
			if not self._deps_done(op):
				self.ready_queue.rotate(-1)
				continue
			if buffer_check is not None and not buffer_check(op):
				self.ready_queue.rotate(-1)
				continue
			if isinstance(op, (MatrixMultOp, MatrixMultTiledOp, Conv1DOp, Conv1DTiledOp, NonLinearityOp, NonLinearityTiledOp)) and op.group_id is None:
				for group_id in range(len(accelerator.groups)):
					if accelerator.group_mac_lanes_free(group_id) > 0:
						op.group_id = group_id
						break
			if isinstance(op, (SoftmaxOp, SoftmaxTiledOp)) and op.group_id is None:
				for group_id in range(len(accelerator.softmax_units)):
					group_softmax = accelerator.softmax_units[group_id]
					if any(unit.ready for unit in group_softmax):
						op.group_id = group_id
						break
			if accelerator.can_assign([op]):
				assigned = accelerator.assign_op(op)
				if assigned:
					self.ready_queue.popleft()
					self.pending.discard(op)
					self.inflight.add(op)
					assigned_ops.append(op)
				else:
					self.ready_queue.rotate(-1)
			else:
				self.ready_queue.rotate(-1)
		return assigned_ops

	def all_done(self):
		return len(self.done) == len(self.dependencies)
