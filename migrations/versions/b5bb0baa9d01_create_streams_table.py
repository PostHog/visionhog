"""create streams table

Revision ID: b5bb0baa9d01
Revises:
Create Date: 2025-05-14 16:00:22.114432

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5bb0baa9d01'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create streams table
    op.create_table(
        'streams',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team', sa.String(), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('emit_events', sa.Boolean(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop streams table
    op.drop_table('streams')
